"""
Muon optimizer adapted from nanochat (which adapted it from modded-nanogpt).
https://github.com/KellerJordan/modded-nanogpt
https://github.com/karpathy/nanochat

Background:
Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
zero even beyond the point where the iteration no longer converges all the way to one everywhere
on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
performance at all relative to UV^T, where USV^T = G is the SVD.

Here, an alternative to Newton-Schulz iteration with potentially better convergence properties:
Polar Express Sign Method for orthogonalization.
https://arxiv.org/pdf/2505.16932
by Noah Amsel, David Persson, Christopher Musco, Robert M. Gower.

Some of the changes in nanochat implementation:
- Uses a simpler, more general approach to parameter grouping and stacking
- Uses a single fused kernel for the momentum -> polar_express -> variance_reduction -> update step
- Makes no assumptions about model architecture (e.g. that attention weights are fused into QKVO format)
"""

import torch
import torch.distributed as dist
from torch import Tensor

# Coefficients for Polar Express (computed for num_iters=5, safety_factor=2e-2, cushion=2)
# From https://arxiv.org/pdf/2505.16932
# These are precomputed (a, b, c) tuples for the iterative orthogonalization formula:
# X_{k+1} = a*X_k + (b*A + c*A^2)@X_k where A = X_k @ X_k^T
# Each iteration brings X closer to an orthogonal matrix
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(
    stacked_grads: Tensor,
    stacked_params: Tensor,
    momentum_buffer: Tensor,
    second_momentum_buffer: Tensor,
    momentum_t: Tensor,
    lr_t: Tensor,
    wd_t: Tensor,
    beta2_t: Tensor,
    ns_steps: int,
    red_dim: int,
) -> None:
    """
    Fused Muon step: momentum -> polar_express -> variance_reduction -> cautious_update
    All in one compiled graph to eliminate Python overhead between ops.
    Some of the constants are 0-D CPU tensors to avoid recompilation when values change.
    """

    # ==================== Step 1: Nesterov Momentum ====================
    # Update the momentum buffer with a weighted average of current gradient
    # momentum_buffer = momentum * momentum_buffer + (1 - momentum) * grad
    # Then compute Nesterov-style update: g = (1-momentum)*grad + momentum*momentum_buffer
    #
    # This expands to: g = G(1 - M²) + M² * V_old
    # With M=0.95: ~10% current gradient, ~90% accumulated history
    #
    # NOTE: .float() cast on lerp_ weight parameter is required for torch.compile with inductor backend
    # in PyTorch 2.6+. The 'weight' parameter in lerp_(x, weight) must be float32, even when tensors are bfloat16.
    # Performance impact: Negligible (~0 overhead, just scalar dtype conversions, not on critical path).
    # To avoid .float(): (1) Don't use torch.compile, OR (2) Use eager mode with torch._dynamo.config.suppress_errors=True,
    # OR (3) Wait for PyTorch to relax this dtype restriction in future versions.
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(
        stacked_grads, (1 - momentum).float()
    )  # a.lerp_(b, weight) = a*(1 - weight) + b*weight
    g = stacked_grads.lerp_(momentum_buffer, momentum.float())  # Nesterov blend

    # ==================== Step 2: Polar Express Orthogonalization ====================
    # Orthogonalize the gradient matrix using iterative Polar Express method.
    # This ensures the update direction has orthonormal columns/rows, which:
    # - Reduces correlations between gradient components
    # - Improves conditioning of the optimization landscape
    # - Provides implicit regularization toward well-conditioned transformations
    X = g.bfloat16()  # Convert to bfloat16 for efficiency

    # Work with the narrow dimension: if tall matrix (rows > cols), transpose
    if g.size(-2) > g.size(-1):
        X = X.mT  # Work with wide matrix (fewer iterations needed)

    # Normalize: divide by norm to bring values into a stable range
    # The 1.02 factor provides slight over-normalization for stability
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)

    # Apply ns_steps iterations of the Polar Express formula
    # Each iteration: X_{k+1} = a*X_k + (b*A + c*A^2)@X_k where A = X_k @ X_k^T
    for a, b, c in polar_express_coeffs[:ns_steps]:
        A = X @ X.mT  # Gram matrix: measures how far X is from orthogonal
        B = b * A + c * (A @ A)  # Polynomial correction term
        X = a * X + B @ X  # Update X to be closer to orthogonal

    # Transpose back if we transposed earlier
    if g.size(-2) > g.size(-1):
        X = X.mT
    g = X  # Now g is approximately orthogonalized

    # ==================== Step 3: Variance Reduction (Adaptive Learning Rate) ====================
    # Similar to Adam's adaptive learning rate, but factored per-row or per-column.
    # This scales the update based on the historical variance of gradients while:
    # - Saving memory (factored representation instead of full matrix)
    # - Preserving gradient norm (unlike standard Adam)
    beta2 = beta2_t.to(g.dtype)

    # Compute variance along the reduction dimension (either rows or cols)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)

    # Calculate the norm of the current gradient
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()

    # Update exponential moving average of variance (like Adam's second moment)
    # NOTE: .float() required for torch.compile (PyTorch 2.6+) - see comment in Step 1 above
    second_momentum_buffer.lerp_(
        v_mean.to(dtype=second_momentum_buffer.dtype), (1 - beta2).float()
    )

    # Compute adaptive step size (inverse sqrt of variance, similar to Adam)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()  # rsqrt = 1/sqrt(x)

    # Calculate what the norm would be after scaling
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()

    # Final scaling factor preserves the gradient norm while adapting per-row/col
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)  # Apply adaptive scaling

    # ==================== Step 4: Cautious Weight Decay + Parameter Update ====================
    # "Cautious" means only apply weight decay where gradient and parameter agree in sign.
    # This prevents weight decay from fighting against the gradient direction, which can
    # slow down training or cause instability.
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)

    # Create mask: True where gradient and parameter have same sign
    mask = (g * stacked_params) >= 0

    # Update parameters: param -= lr * gradient + lr * weight_decay * param * mask
    # Weight decay only applied where mask is True (gradient and param agree in sign)
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


# ==============================================================================
# Combined MuonAdamW Optimizer (Nanochat-style)
# ==============================================================================
# This combines both AdamW and Muon into a single optimizer with unified
# parameter groups. Enables nanochat-style training without DDP wrapper.


@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(
    p: Tensor,  # parameter tensor
    grad: Tensor,  # gradient, same shape as p
    exp_avg: Tensor,  # first moment, same shape as p
    exp_avg_sq: Tensor,  # second moment, same shape as p
    step_t: Tensor,  # 0-D CPU tensor, step count
    lr_t: Tensor,  # 0-D CPU tensor, learning rate
    beta1_t: Tensor,  # 0-D CPU tensor, beta1
    beta2_t: Tensor,  # 0-D CPU tensor, beta2
    eps_t: Tensor,  # 0-D CPU tensor, epsilon
    wd_t: Tensor,  # 0-D CPU tensor, weight decay
) -> None:
    """
    Fused AdamW step: weight_decay -> momentum_update -> bias_correction -> param_update
    All in one compiled graph to eliminate Python overhead between ops.
    The 0-D CPU tensors avoid recompilation when hyperparameter values change.
    """
    # Weight decay (decoupled, applied before the update)
    p.mul_(1 - lr_t * wd_t)
    # Update running averages (lerp_ is cleaner and fuses well)
    # NOTE: .float() required for torch.compile (PyTorch 2.6+) - lerp_ weight param must be float32
    exp_avg.lerp_(grad, (1 - beta1_t).float())
    exp_avg_sq.lerp_(grad.square(), (1 - beta2_t).float())
    # Bias corrections
    bias1 = 1 - beta1_t**step_t
    bias2 = 1 - beta2_t**step_t
    # Compute update and apply
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)


class MuonAdamW(torch.optim.Optimizer):
    """
    Combined optimizer: Muon for 2D matrix params, AdamW for others (single GPU).

    This is the nanochat-style combined optimizer that handles both parameter
    types in a single optimizer with unified parameter groups.

    Arguments:
        param_groups: List of dicts, each containing:
            - 'params': List of parameters
            - 'kind': 'adamw' or 'muon'
            - For AdamW groups: 'lr', 'betas', 'eps', 'weight_decay'
            - For Muon groups: 'lr', 'momentum', 'ns_steps', 'beta2', 'weight_decay'
    """

    def __init__(self, param_groups: list[dict]):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        # AdamW tensors
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        # Muon tensors
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group: dict) -> None:
        """AdamW update for each param in the group individually."""
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]

            # State init
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            state["step"] += 1

            # Fill 0-D tensors with current values
            self._adamw_step_t.fill_(state["step"])
            self._adamw_lr_t.fill_(group["lr"])
            self._adamw_beta1_t.fill_(group["betas"][0])
            self._adamw_beta2_t.fill_(group["betas"][1])
            self._adamw_eps_t.fill_(group["eps"])
            self._adamw_wd_t.fill_(group["weight_decay"])

            # Fused update
            adamw_step_fused(
                p,
                grad,
                exp_avg,
                exp_avg_sq,
                self._adamw_step_t,
                self._adamw_lr_t,
                self._adamw_beta1_t,
                self._adamw_beta2_t,
                self._adamw_eps_t,
                self._adamw_wd_t,
            )

    def _step_muon(self, group: dict) -> None:
        """Muon update for all params in the group (stacked for efficiency)."""
        params: list[Tensor] = group["params"]
        if not params:
            return

        # Get or create group-level buffers
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype

        # Momentum buffer
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(
                num_params, *shape, dtype=dtype, device=device
            )
        momentum_buffer = state["momentum_buffer"]

        # Second momentum buffer (factored)
        if "second_momentum_buffer" not in state:
            state_shape = (
                (num_params, shape[-2], 1)
                if shape[-2] >= shape[-1]
                else (num_params, 1, shape[-1])
            )
            state["second_momentum_buffer"] = torch.zeros(
                state_shape, dtype=dtype, device=device
            )
        second_momentum_buffer = state["second_momentum_buffer"]
        red_dim = -1 if shape[-2] >= shape[-1] else -2

        # Validate all parameters have gradients before stacking
        assert all(
            p.grad is not None for p in params
        ), "All parameters in Muon group must have gradients. Check if all parameters are used in forward pass."

        # Stack grads and params
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)

        # Fill 0-D tensors
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
        self._muon_wd_t.fill_(group["weight_decay"])

        # Single fused kernel
        muon_step_fused(
            stacked_grads,
            stacked_params,
            momentum_buffer,
            second_momentum_buffer,
            self._muon_momentum_t,
            self._muon_lr_t,
            self._muon_wd_t,
            self._muon_beta2_t,
            group["ns_steps"],
            red_dim,
        )

        # Copy back to original params
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group["kind"] == "adamw":
                self._step_adamw(group)
            elif group["kind"] == "muon":
                self._step_muon(group)
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")


class DistMuonAdamW(torch.optim.Optimizer):
    """
    Combined distributed optimizer: Muon for 2D matrix params, AdamW for others.

    This is the nanochat-style optimizer that enables distributed training
    WITHOUT using PyTorch's DDP wrapper. Gradient synchronization happens
    inside the optimizer using manual reduce_scatter/all_gather operations.

    Design Goals:
    - Overlap communication with computation (async ops)
    - Minimize memory by sharding optimizer states across ranks (ZeRO-2 style)
    - No DDP wrapper needed - model stays unwrapped

    Communication Pattern (3-phase async):
        Phase 1: Launch all async reduce ops (reduce_scatter/all_reduce)
        Phase 2: Wait for reduces, compute updates, launch gathers
        Phase 3: Wait for gathers, copy back

    Arguments:
        param_groups: List of dicts, each containing:
            - 'params': List of parameters
            - 'kind': 'adamw' or 'muon'
            - For AdamW groups: 'lr', 'betas', 'eps', 'weight_decay'
            - For Muon groups: 'lr', 'momentum', 'ns_steps', 'beta2', 'weight_decay'
    """

    def __init__(self, param_groups: list[dict]):
        super().__init__(param_groups, defaults={})

        # Create 0-D CPU tensors for all hyperparameters
        #
        # Why 0-D CPU tensors?
        # - torch.compile creates a compiled graph based on input shapes and dtypes
        # - If we pass Python floats as arguments, changing them triggers recompilation
        # - 0-D tensors act as "symbolic" values: we can change their contents without
        #   triggering recompilation (the compiled graph sees "a tensor" not "the value 0.001")
        # - CPU location means no GPU->CPU transfer overhead when filling values
        #
        # Example: If lr changes from 0.001 to 0.0005 during training:
        # - With Python float: graph recompiles (slow, ~1-2 seconds)
        # - With 0-D tensor: just fill new value (fast, microseconds)

        # AdamW hyperparameter tensors
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

        # Muon hyperparameter tensors
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _reduce_adamw(self, group: dict, world_size: int) -> dict:
        """
        Launch async reduce ops for AdamW group.

        This is PHASE 1 for AdamW parameters: we initiate gradient synchronization across
        all GPUs but don't wait for completion (async_op=True). This allows computation
        and communication to overlap for better GPU utilization.

        Two strategies based on parameter size:
        1. Small params (<1024 elements): all_reduce
           - Every GPU gets a complete copy of the averaged gradient
           - Simple but requires all GPUs to store full gradient
           - Good for embeddings, biases, layer norms

        2. Large params (>=1024 elements): reduce_scatter
           - Each GPU only gets its slice of the averaged gradient
           - Memory-efficient (ZeRO-2 style sharding)
           - Each GPU will update only its slice of parameters
           - Good for large weight matrices
        """
        param_infos = {}
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad

            if p.numel() < 1024:
                # Small params: all_reduce (every GPU gets the complete averaged gradient)
                # op=ReduceOp.AVG means gradients from all GPUs are averaged (not summed)
                # async_op=True returns immediately without blocking - comm happens in background
                future = dist.all_reduce(
                    grad, op=dist.ReduceOp.AVG, async_op=True
                ).get_future()
                param_infos[p] = dict(future=future, grad_slice=grad, is_small=True)
            else:
                # Large params: reduce_scatter (memory efficient - each GPU gets only its slice)
                # Split parameter along first dimension: each GPU will own rank_size elements
                rank_size = grad.shape[0] // world_size
                # grad_slice will receive the averaged gradient for this GPU's slice
                grad_slice = torch.empty_like(grad[:rank_size])
                # reduce_scatter: all GPUs contribute their full gradient, but each receives
                # only their slice of the averaged result. This saves memory!
                future = dist.reduce_scatter_tensor(
                    grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True
                ).get_future()
                param_infos[p] = dict(
                    future=future, grad_slice=grad_slice, is_small=False
                )
        return dict(param_infos=param_infos)

    def _reduce_muon(self, group: dict, world_size: int) -> dict:
        """
        Launch async reduce op for Muon group.

        This is PHASE 1 for Muon parameters (2D weight matrices). Unlike AdamW which
        processes each parameter individually, Muon works on stacked parameters for
        efficiency (one big fused kernel instead of many small ones).

        Key concept: SHARDING
        - We divide the group's parameters equally among all GPUs
        - Each GPU will be responsible for updating its chunk
        - Example: 100 weight matrices across 4 GPUs -> each GPU owns 25 matrices

        Steps:
        1. Stack all gradients into a single tensor: [num_params, ...shape]
        2. Pad to ensure even division across GPUs (pad with zeros if needed)
        3. reduce_scatter: Each GPU gets the averaged gradient for its chunk only
        """
        params = group["params"]
        # Calculate how many parameters each GPU will own (round up to ensure coverage)
        chunk_size = (len(params) + world_size - 1) // world_size
        # Total after padding (must be divisible by world_size for even splitting)
        padded_num_params = chunk_size * world_size
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype

        # Stack all gradients into one big tensor: [num_params, height, width]
        # This allows us to process all parameters together in a single fused kernel
        grad_stack = torch.stack([p.grad for p in params if p.grad is not None])
        stacked_grads = torch.empty(
            padded_num_params, *shape, dtype=dtype, device=device
        )
        stacked_grads[: len(params)].copy_(grad_stack)
        # Zero-pad if needed (e.g., 100 params across 4 GPUs -> pad to 100 with zeros)
        # The padding ensures even chunks: chunk_size=25, so we get 4 chunks of 25
        if len(params) < padded_num_params:
            stacked_grads[len(params) :].zero_()

        # Prepare buffer for this GPU's chunk of averaged gradients
        # After reduce_scatter, each GPU will have chunk_size averaged gradients
        grad_chunk = torch.empty(chunk_size, *shape, dtype=dtype, device=device)
        # reduce_scatter: All GPUs send their full stacked_grads, but each receives
        # only their chunk of the averaged result. This distributes the workload!
        # GPU 0 gets chunks [0:chunk_size], GPU 1 gets [chunk_size:2*chunk_size], etc.
        future = dist.reduce_scatter_tensor(
            grad_chunk, stacked_grads, op=dist.ReduceOp.AVG, async_op=True
        ).get_future()

        return dict(
            future=future,
            grad_chunk=grad_chunk,
            stacked_grads=stacked_grads,  # Will reuse this buffer for all_gather later
            chunk_size=chunk_size,
        )

    def _compute_adamw(
        self, group: dict, info: dict, gather_list: list, rank: int, world_size: int
    ) -> None:
        """
        Wait for reduce, compute AdamW updates, launch gathers for large params.

        This is PHASE 2 for AdamW: Now that gradient synchronization is complete,
        we can compute parameter updates. The strategy differs based on param size:

        Small params:
        - All GPUs have the full gradient (from all_reduce in Phase 1)
        - All GPUs update the full parameter independently
        - All GPUs end up with identical parameter values
        - No gather needed since everyone has the complete param

        Large params (memory efficient):
        - Each GPU has only its slice of the gradient (from reduce_scatter in Phase 1)
        - Each GPU updates only its slice of the parameter
        - Optimizer state (momentum, variance) is also sharded - only for this slice
        - After update, we launch all_gather to share the updated slices
        - This is ZeRO-2 style: sharded optimizer states, but full model parameters
        """
        param_infos = info["param_infos"]
        for p in group["params"]:
            if p not in param_infos:
                continue
            pinfo = param_infos[p]
            # Wait for the gradient synchronization to complete (from Phase 1)
            pinfo["future"].wait()
            grad_slice = pinfo["grad_slice"]
            state = self.state[p]

            # Determine which portion of the parameter this GPU will update
            if pinfo["is_small"]:
                # Small param: operate on the entire parameter (all GPUs do this)
                p_slice = p
            else:
                # Large param: operate only on this GPU's slice
                # Example: 4 GPUs, parameter shape [1024, 768]
                # rank_size = 256, so GPU 0 owns [0:256], GPU 1 owns [256:512], etc.
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size : (rank + 1) * rank_size]

            # Initialize optimizer state for this parameter (only for the slice we own)
            # This is key to memory efficiency: we only store momentum/variance for our slice
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p_slice)  # First moment (momentum)
                state["exp_avg_sq"] = torch.zeros_like(
                    p_slice
                )  # Second moment (variance)
            state["step"] += 1

            # Fill 0-D tensors with current hyperparameters
            # (these are 0-D CPU tensors to avoid recompilation when values change)
            self._adamw_step_t.fill_(state["step"])
            self._adamw_lr_t.fill_(group["lr"])
            self._adamw_beta1_t.fill_(group["betas"][0])
            self._adamw_beta2_t.fill_(group["betas"][1])
            self._adamw_eps_t.fill_(group["eps"])
            self._adamw_wd_t.fill_(group["weight_decay"])

            # Run the fused AdamW kernel to update the parameter slice
            adamw_step_fused(
                p_slice,
                grad_slice,
                state["exp_avg"],
                state["exp_avg_sq"],
                self._adamw_step_t,
                self._adamw_lr_t,
                self._adamw_beta1_t,
                self._adamw_beta2_t,
                self._adamw_eps_t,
                self._adamw_wd_t,
            )

            # Large params need all_gather to reconstruct the full parameter
            # Small params already have the full parameter on all GPUs, so skip this
            if not pinfo["is_small"]:
                # all_gather: Each GPU broadcasts its updated slice to all other GPUs
                # Result: all GPUs have the complete updated parameter
                future = dist.all_gather_into_tensor(
                    p, p_slice, async_op=True
                ).get_future()
                gather_list.append(dict(future=future, params=None))

    def _compute_muon(
        self, group: dict, info: dict, gather_list: list, rank: int
    ) -> None:
        """
        Wait for reduce, compute Muon updates, launch gather.

        This is PHASE 2 for Muon: Compute updates for the parameters this GPU owns.

        Key memory optimization: SHARDED OPTIMIZER STATE
        - Each GPU only stores momentum/variance for the parameters it owns
        - Example: 100 weight matrices, 4 GPUs -> each GPU stores state for 25 matrices
        - This is ZeRO-2 style optimization: sharded optimizer states

        Steps:
        1. Wait for gradient synchronization (reduce_scatter from Phase 1)
        2. Determine which parameters this GPU owns (based on rank and chunk_size)
        3. Create/retrieve sharded optimizer state (only for owned parameters)
        4. Run the fused Muon update kernel on owned parameters
        5. Launch all_gather to share updated parameters with all GPUs
        """
        # Wait for gradient synchronization to complete (from Phase 1)
        info["future"].wait()
        params = group["params"]
        chunk_size = info["chunk_size"]
        grad_chunk = info["grad_chunk"]  # This GPU's chunk of averaged gradients
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype

        # Determine which parameters this GPU owns
        # Example: 100 params, 4 GPUs, chunk_size=25
        # rank 0: owns params [0:25], rank 1: [25:50], rank 2: [50:75], rank 3: [75:100]
        start_idx = rank * chunk_size
        # Handle uneven division: last GPU might own fewer params
        # e.g., 100 params / 4 GPUs = 25, but if we had 97 params, rank 3 would own 22
        num_owned = min(chunk_size, max(0, len(params) - start_idx))

        # Get or create group-level optimizer state (SHARDED - only for our chunk)
        # Note: state is keyed by first param, but applies to entire group
        state = self.state[p]

        # Initialize momentum buffer (first moment) - shape: [chunk_size, height, width]
        # Only allocated for chunk_size parameters, not all parameters!
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(
                chunk_size, *shape, dtype=dtype, device=device
            )

        # Initialize second momentum buffer (adaptive learning rate) - FACTORED representation
        # Instead of storing full [chunk_size, height, width], we store:
        # - [chunk_size, height, 1] if height >= width (reduce along width dimension)
        # - [chunk_size, 1, width] if width > height (reduce along height dimension)
        # This saves memory while preserving per-row or per-column adaptation
        if "second_momentum_buffer" not in state:
            state_shape = (
                (chunk_size, shape[-2], 1)
                if shape[-2] >= shape[-1]
                else (chunk_size, 1, shape[-1])
            )
            state["second_momentum_buffer"] = torch.zeros(
                state_shape, dtype=dtype, device=device
            )
        # Determine which dimension we're reducing over (for factored variance)
        red_dim = -1 if shape[-2] >= shape[-1] else -2

        # Prepare output buffer for the updated parameters (will all_gather this)
        updated_params = torch.empty(chunk_size, *shape, dtype=dtype, device=device)

        if num_owned > 0:
            # Get the parameters this GPU actually owns (not the padding)
            owned_params = [params[start_idx + i] for i in range(num_owned)]
            stacked_owned = torch.stack(owned_params)

            # Fill 0-D tensors with current hyperparameters
            self._muon_momentum_t.fill_(group["momentum"])
            self._muon_beta2_t.fill_(group["beta2"])
            # Note: Muon uses aspect-ratio-adjusted learning rate
            # Taller matrices (more rows) get higher lr by sqrt(aspect_ratio)
            self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
            self._muon_wd_t.fill_(group["weight_decay"])

            # Run the fused Muon kernel (momentum + orthogonalization + variance reduction + update)
            muon_step_fused(
                grad_chunk[:num_owned],  # Only process gradients for params we own
                stacked_owned,  # The actual parameters we own
                state["momentum_buffer"][:num_owned],  # Our slice of momentum buffer
                state["second_momentum_buffer"][
                    :num_owned
                ],  # Our slice of variance buffer
                self._muon_momentum_t,
                self._muon_lr_t,
                self._muon_wd_t,
                self._muon_beta2_t,
                group[
                    "ns_steps"
                ],  # Number of Newton-Schulz iterations for orthogonalization
                red_dim,
            )
            # Copy updated parameters into the output buffer
            updated_params[:num_owned].copy_(stacked_owned)

        # Zero-pad if this GPU owns fewer than chunk_size params (for last GPU)
        if num_owned < chunk_size:
            updated_params[num_owned:].zero_()

        # Launch all_gather to collect updated parameters from all GPUs
        # We reuse the stacked_grads buffer (no longer needed) to save memory
        stacked_params = info["stacked_grads"]
        # all_gather: Each GPU sends its updated_params chunk, and everyone receives
        # the concatenated result from all GPUs. After this, all GPUs have all parameters.
        future = dist.all_gather_into_tensor(
            stacked_params, updated_params, async_op=True
        ).get_future()
        gather_list.append(
            dict(future=future, stacked_params=stacked_params, params=params)
        )

    def _finish_gathers(self, gather_list: list) -> None:
        """
        Wait for all gathers and copy Muon params back.

        This is PHASE 3: Finalize parameter synchronization.

        At this point, all async all_gather operations from Phase 2 have been launched.
        Now we wait for them to complete and copy the results back to the original
        parameter tensors.

        For AdamW large params:
        - The all_gather wrote directly into the parameter tensor (in-place)
        - Nothing to copy back (params=None in gather_list)

        For Muon params:
        - all_gather wrote into stacked_params buffer
        - We need to unstack and copy each parameter back to its original location
        - This ensures the model's parameters are updated and ready for next forward pass
        """
        for info in gather_list:
            # Wait for the all_gather operation to complete
            info["future"].wait()
            if info["params"] is not None:
                # Muon case: copy from stacked buffer back to individual parameter tensors
                # stacked_params is shape [num_params, height, width]
                # We unbind along dim 0 to get individual [height, width] tensors
                # and copy each back to its corresponding parameter
                torch._foreach_copy_(
                    info["params"],
                    list(info["stacked_params"][: len(info["params"])].unbind(0)),
                )

    @torch.no_grad()
    def step(self):
        """
        Execute one optimization step with 3-phase asynchronous communication.

        This is the main entry point that orchestrates distributed parameter updates.
        The 3-phase design maximizes overlap between communication and computation:

        PHASE 1: Launch gradient synchronization (non-blocking)
        ├─ AdamW: all_reduce (small) or reduce_scatter (large)
        ├─ Muon: reduce_scatter (always sharded)
        └─ Returns immediately, communication happens in background

        PHASE 2: Compute updates and launch parameter synchronization (non-blocking)
        ├─ Wait for Phase 1 communication to complete
        ├─ Compute parameter updates using synchronized gradients
        ├─ For large/sharded params: launch all_gather to share updated slices
        └─ Returns immediately, communication happens in background

        PHASE 3: Finalize
        ├─ Wait for Phase 2 communication to complete
        └─ Copy results back to model parameters (Muon only)

        Why 3 phases?
        - Without async: wait for reduce -> compute -> wait for gather (sequential)
        - With async: launch all reduces -> (they run in parallel) -> wait and compute
          -> launch all gathers -> (they run in parallel) -> finalize
        - This overlaps communication across different parameter groups and with computation
        - Can improve training throughput by 10-30% compared to synchronous approach

        Memory efficiency (ZeRO-2 style):
        - Large params: optimizer states (momentum, variance) are sharded across GPUs
        - Small params: replicated on all GPUs for simplicity
        - Model parameters: replicated (all GPUs have full model after step)
        - Example: 4 GPUs, 1B params -> ~250M optimizer state per GPU vs 1B without sharding
        """
        rank = dist.get_rank()  # This GPU's ID (0, 1, 2, ... world_size-1)
        world_size = dist.get_world_size()  # Total number of GPUs

        # ==================== PHASE 1: Launch async gradient synchronization ====================
        # Launch all reduce operations without waiting for them to complete
        # This allows all communication to happen in parallel across parameter groups
        reduce_infos: list[dict] = []
        for group in self.param_groups:
            if group["kind"] == "adamw":
                reduce_infos.append(self._reduce_adamw(group, world_size))
            elif group["kind"] == "muon":
                reduce_infos.append(self._reduce_muon(group, world_size))
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")

        # ==================== PHASE 2: Compute updates and launch parameter sync ====================
        # Now we wait for each group's reduce to complete, compute its update,
        # and immediately launch the gather (if needed) before moving to next group
        gather_list: list[dict] = []
        for group, info in zip(self.param_groups, reduce_infos):
            if group["kind"] == "adamw":
                self._compute_adamw(group, info, gather_list, rank, world_size)
            elif group["kind"] == "muon":
                self._compute_muon(group, info, gather_list, rank)
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")

        # ==================== PHASE 3: Finalize parameter synchronization ====================
        # Wait for all parameter gathers to complete and copy results back
        self._finish_gathers(gather_list)
