import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False
        )
        
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialize special params
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt_proj bias
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in real space
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_reinit = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_reinit = True

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seq_len, dim = hidden_states.shape

        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1) # (B, L, d_inner)

        # Conv step
        x = x.permute(0, 2, 1) # (B, d_inner, L)
        x = self.conv1d(x)[:, :, :seq_len] # Depthwise conv
        x = x.permute(0, 2, 1) # (B, L, d_inner)
        x = F.silu(x)

        # SSM step
        # A, D are parameters. B, C, dt are input-dependent.
        
        # Project to custom parameters
        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2*d_state)
        
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        dt = self.dt_proj(dt) # (B, L, d_inner)
        
        # Parameter discretization
        dt = F.softplus(dt) # (B, L, d_inner)
        A = -torch.exp(self.A_log) # (d_inner, d_state)
        
        # Selective Scan involves a recurrence. 
        # y_t = SSM(A, B_t, C_t, dt_t, x_t)
        # This is the slow part in Python.
        
        y = self.selective_scan(x, dt, A, B, C, self.D)
        
        y = y * F.silu(z)
        out = self.out_proj(y)
        
        return out

    def selective_scan(self, u, dt, A, B, C, D):
        """
        u: (B, L, d_inner)
        dt: (B, L, d_inner)
        A: (d_inner, d_state)
        B: (B, L, d_state)
        C: (B, L, d_state)
        D: (d_inner)
        """
        batch_size, seq_len, d_inner = u.shape
        d_state = A.shape[1]
        
        # Discretize A: dA = exp(dt * A)
        # However, A is (D, N) and dt is (B, L, D). 
        # We need dA per step.
        # dA = torch.exp(torch.einsum("bld,dn->bldn", dt, A)) 
        
        # Discretize B: dB = dt * B
        # B is (B, L, N), dt is (B, L, D). 
        # deltaB_u = (dt * u) * B ? No.
        
        # Let's use the parallel scan formulation (associative scan) or simple recurrence.
        # Simple recurrence is easiest to implement correctly without custom kernels.
        
        # dA = exp(dt * A) 
        # x_t = dA * x_{t-1} + dt * B_t * u_t
        # y_t = C_t * x_t + D * u_t
        
        # We perform this in loop for clarity/correctness (slow but functional).
        
        deltaA = torch.exp(torch.einsum("bld,dn->bldn", dt, A)) # (B, L, D, N)
        deltaB_u = torch.einsum("bld,bln,bld->bldn", dt, B, u) # (B, L, D, N)
        
        # Scan
        # h_t = deltaA_t * h_{t-1} + deltaB_u_t
        # h dim: (B, D, N)
        
        x = torch.zeros((batch_size, d_inner, d_state), device=u.device)
        ys = []
        
        for i in range(seq_len):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            # y_t = C_t * x_t
            # C_t: (B, N) -> (B, 1, N) for broadcast
            # x: (B, D, N)
            
            y_t = torch.einsum("bdn,bn->bd", x, C[:, i])
            ys.append(y_t)
            
        y = torch.stack(ys, dim=1) # (B, L, D)
        
        # Add skip connection D * u
        y = y + u * D
        
        return y
