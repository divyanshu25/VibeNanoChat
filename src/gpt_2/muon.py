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
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)  # lerp(a, b, w) = a + w*(b - a)
    g = stacked_grads.lerp_(
        momentum_buffer, momentum
    )  # Blend gradient with momentum buffer

    # ==================== Step 2: Polar Express Orthogonalization ====================
    # Orthogonalize the gradient matrix using iterative Polar Express method
    # This ensures the update direction has orthonormal columns/rows
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
    # Similar to Adam's adaptive learning rate, but factored per-row or per-column
    # This scales the update based on the historical variance of gradients
    beta2 = beta2_t.to(g.dtype)

    # Compute variance along the reduction dimension (either rows or cols)
    v_mean = (
        g.float().square().mean(dim=red_dim, keepdim=True)
    )  # Mean of squared values
    red_dim_size = g.size(red_dim)  # Size of the reduction dimension

    # Calculate the norm of the current gradient
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()

    # Update exponential moving average of variance (like Adam's second moment)
    second_momentum_buffer.lerp_(
        v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2
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
    # "Cautious" means only apply weight decay where gradient and parameter agree in sign
    # This prevents weight decay from fighting against the gradient direction
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)

    # Create mask: True where gradient and parameter have same sign
    # This identifies parameters where weight decay would move in same direction as gradient
    mask = (g * stacked_params) >= 0

    # Update parameters: param -= lr * gradient + lr * weight_decay * param * mask
    # Weight decay only applied where mask is True (gradient and param agree in sign)
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        ns_steps: The number of Newton-Schulz iteration steps to use.
        beta2: The decay rate for the second moment (variance) estimate. Set to None to disable.
        weight_decay: Cautious weight decay coefficient. Only decays where update and weight agree.
    """

    def __init__(
        self, params, lr=0.02, momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=0.0
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            ns_steps=ns_steps,
            beta2=beta2,
            weight_decay=weight_decay,
        )

        # Muon only works with 2D parameters (weight matrices)
        assert all(p.ndim == 2 for p in params), "Muon expects 2D parameters only"
        params = list(
            params
        )  # Ensure we have a list, not an e.g. (exhaustible) iterator

        # Group parameters by shape so we can stack tensors efficiently
        # Stacking allows batched operations across multiple parameters in a single kernel
        shapes = sorted(
            {p.shape for p in params}
        )  # Get unique shapes, sorted for determinism
        param_groups = []
        for shape in shapes:
            # All parameters with the same shape go into one group
            group_params = [p for p in params if p.shape == shape]
            param_groups.append(dict(params=group_params))
        super().__init__(param_groups, defaults)

        # Store hyperparameters as 0-D CPU tensors to avoid torch.compile recompilation
        # When values change, torch.compile sees the tensor reference is the same, not a new Python float
        self._momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    @torch.no_grad()
    def step(self):
        # Process each parameter group (parameters are grouped by shape)
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            if not params:
                continue

            # ==================== Initialize State Buffers ====================
            # Store optimizer state in the first parameter's state dict (for convenience)
            state = self.state[params[0]]
            num_params = len(
                params
            )  # e.g.: 12 (for a 12-layer model with same-shaped weights)
            # e.g.: shape = (768, 3072), device = cuda:0, dtype = torch.float32
            shape, device, dtype = params[0].shape, params[0].device, params[0].dtype

            # Create first momentum buffer (full size for each parameter)
            # This stores the momentum-averaged gradients
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros(
                    num_params, *shape, dtype=dtype, device=device
                )
            momentum_buffer = state["momentum_buffer"]  # e.g.: (12, 768, 3072)

            # Create second momentum buffer (factored to save memory)
            # This stores variance estimates, but only per-row OR per-column (not full matrix)
            if "second_momentum_buffer" not in state:
                if shape[-2] >= shape[-1]:
                    # Tall or square matrix: track variance per row
                    state["second_momentum_buffer"] = torch.zeros(
                        num_params, shape[-2], 1, dtype=dtype, device=device
                    )
                else:
                    # Wide matrix: track variance per column
                    state["second_momentum_buffer"] = torch.zeros(
                        num_params, 1, shape[-1], dtype=dtype, device=device
                    )
            second_momentum_buffer = state[
                "second_momentum_buffer"
            ]  # e.g.: (12, 768, 1)
            red_dim = (
                -1 if shape[-2] >= shape[-1] else -2
            )  # Which dimension to reduce over

            # ==================== Stack Parameters for Batched Processing ====================
            # Stack all parameters and gradients into single tensors
            # This allows processing all parameters in one fused kernel call
            stacked_grads = torch.stack([p.grad for p in params])  # (12, 768, 3072)
            stacked_params = torch.stack(params)  # (12, 768, 3072)

            # ==================== Fill Hyperparameter Tensors ====================
            # Update the 0-D CPU tensors with current hyperparameter values
            self._momentum_t.fill_(group["momentum"])
            self._beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
            # Scale learning rate based on aspect ratio (helps with wide/tall matrices)
            self._lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
            self._wd_t.fill_(group["weight_decay"])

            # ==================== Execute Fused Update ====================
            # Single compiled kernel does: momentum -> orthogonalization -> variance_reduction -> update
            muon_step_fused(
                stacked_grads,
                stacked_params,
                momentum_buffer,
                second_momentum_buffer,
                self._momentum_t,
                self._lr_t,
                self._wd_t,
                self._beta2_t,
                group["ns_steps"],
                red_dim,
            )

            # ==================== Copy Updated Parameters Back ====================
            # Unstack and copy updated parameters back to original parameter tensors
            # torch._foreach_copy_ is an efficient batched copy operation
            torch._foreach_copy_(params, list(stacked_params.unbind(0)))


class DistMuon(torch.optim.Optimizer):
    """
    Distributed version of the Muon optimizer.

    Uses a 3-phase approach:
    1. reduce_scatter: Each rank receives averaged gradients for its parameter chunk
    2. Local update: Each rank updates its owned parameters
    3. all_gather: Broadcast updated parameters back to all ranks

    This distributes both computation and memory across ranks.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        ns_steps: int = 5,
        beta2: float = 0.95,
        weight_decay: float = 0.0,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            ns_steps=ns_steps,
            beta2=beta2,
            weight_decay=weight_decay,
        )

        # Muon only works with 2D parameters (weight matrices)
        assert all(p.ndim == 2 for p in params), "Muon expects 2D parameters only"
        params = list(params)

        # Get distributed training info
        world_size = dist.get_world_size()  # Number of GPUs/processes
        rank = dist.get_rank()  # This process's rank (0 to world_size-1)

        # Group parameters by shape for efficient batched operations
        shapes = sorted(
            {p.shape for p in params}
        )  # Sort for deterministic ordering across ranks
        param_groups = []
        for shape in shapes:
            group_params = [p for p in params if p.shape == shape]
            device, dtype = group_params[0].device, group_params[0].dtype
            # Ensure all params in group have same device/dtype
            assert all(p.device == device for p in group_params)
            assert all(p.dtype == dtype for p in group_params)

            # Compute chunk size: how many parameters each rank will own and update
            # Ceiling division ensures we allocate enough space even if not evenly divisible
            chunk_size = (len(group_params) + world_size - 1) // world_size
            if rank == 0:
                print(
                    f"Muon: {len(group_params)} params of shape {shape}, chunk_size={chunk_size}"
                )
            param_groups.append(dict(params=group_params, chunk_size=chunk_size))
        super().__init__(param_groups, defaults)

        # Store hyperparameters as 0-D CPU tensors to avoid torch.compile recompilation
        self._momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Ensure all grads exist
        assert all(
            p.grad is not None for group in self.param_groups for p in group["params"]
        ), "All params must have grads"

        # ==================== PHASE 1: Stack Gradients and Launch reduce_scatter ====================
        # For each parameter group, stack gradients and start async communication
        # reduce_scatter: Each rank gets the average of one chunk of parameters
        group_infos = []
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            chunk_size = group["chunk_size"]
            padded_num_params = chunk_size * world_size  # Total params after padding
            shape = params[0].shape
            device, dtype = params[0].device, params[0].dtype

            # Stack all gradients into a single tensor (single efficient kernel via torch.stack)
            grad_stack = torch.stack([p.grad for p in params])
            stacked_grads = torch.empty(
                padded_num_params, *shape, dtype=dtype, device=device
            )
            stacked_grads[: len(params)].copy_(grad_stack)

            # Zero-pad if we have fewer params than padded size (for even distribution)
            if len(params) < padded_num_params:
                stacked_grads[len(params) :].zero_()

            # Allocate output buffer for this rank's chunk of averaged gradients
            grad_chunk = torch.empty(chunk_size, *shape, dtype=dtype, device=device)

            # Launch async reduce_scatter:
            # - Each rank sends its stacked_grads to all ranks
            # - Each rank receives averaged gradients for its assigned chunk
            # - async_op=True means this returns immediately and runs in background
            reduce_future = dist.reduce_scatter_tensor(
                grad_chunk, stacked_grads, op=dist.ReduceOp.AVG, async_op=True
            ).get_future()

            # Store info for next phase
            group_infos.append(
                dict(
                    grad_chunk=grad_chunk,  # Will be filled when reduce completes
                    reduce_future=reduce_future,  # Handle to wait on
                    stacked_grads=stacked_grads,  # Reuse this buffer for all_gather output later
                )
            )

        # ==================== PHASE 2: Wait for Reduce, Update Owned Parameters, Launch all_gather ====================
        # Each rank updates only its assigned chunk of parameters, then starts broadcasting results
        all_gather_futures = []
        for group, info in zip(self.param_groups, group_infos):
            # Wait for reduce_scatter to complete - now grad_chunk contains averaged gradients
            info["reduce_future"].wait()

            params = group["params"]
            chunk_size = group["chunk_size"]
            shape = params[0].shape
            device, dtype = params[0].device, params[0].dtype
            grad_chunk = info["grad_chunk"]

            # Determine which parameters this rank owns
            # Example: 12 params, 4 ranks, chunk_size=3
            #   rank 0: params 0-2, rank 1: params 3-5, rank 2: params 6-8, rank 3: params 9-11
            start_idx = rank * chunk_size
            num_owned = min(chunk_size, max(0, len(params) - start_idx))

            # ==================== Initialize State Buffers ====================
            state = self.state[params[0]]

            # First momentum buffer (for momentum-averaged gradients)
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros(
                    chunk_size, *shape, dtype=dtype, device=device
                )
            momentum_buffer = state["momentum_buffer"]

            # Second momentum buffer (factored variance estimates)
            if "second_momentum_buffer" not in state:
                if shape[-2] >= shape[-1]:
                    state["second_momentum_buffer"] = torch.zeros(
                        chunk_size, shape[-2], 1, dtype=dtype, device=device
                    )
                else:
                    state["second_momentum_buffer"] = torch.zeros(
                        chunk_size, 1, shape[-1], dtype=dtype, device=device
                    )
            second_momentum_buffer = state["second_momentum_buffer"]
            red_dim = -1 if shape[-2] >= shape[-1] else -2

            # Allocate buffer for updated parameters (to be gathered across ranks)
            updated_params = torch.empty(chunk_size, *shape, dtype=dtype, device=device)

            # ==================== Update Owned Parameters ====================
            if num_owned > 0:
                # Stack the parameters this rank owns
                owned_params = [params[start_idx + i] for i in range(num_owned)]
                stacked_owned_params = torch.stack(owned_params)

                # Extract slices of buffers/grads corresponding to owned params
                owned_grads = grad_chunk[:num_owned]
                owned_momentum = momentum_buffer[:num_owned]
                owned_second_momentum = second_momentum_buffer[:num_owned]

                # Fill hyperparameter tensors
                self._momentum_t.fill_(group["momentum"])
                self._beta2_t.fill_(
                    group["beta2"] if group["beta2"] is not None else 0.0
                )
                self._lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
                self._wd_t.fill_(group["weight_decay"])

                # Execute fused update on owned parameters only
                muon_step_fused(
                    owned_grads,
                    stacked_owned_params,
                    owned_momentum,
                    owned_second_momentum,
                    self._momentum_t,
                    self._lr_t,
                    self._wd_t,
                    self._beta2_t,
                    group["ns_steps"],
                    red_dim,
                )

                # Copy updated params to output buffer
                updated_params[:num_owned].copy_(stacked_owned_params)

            # Zero-pad the rest (for ranks that own fewer params due to uneven division)
            if num_owned < chunk_size:
                updated_params[num_owned:].zero_()

            # Reuse stacked_grads buffer for all_gather output (memory efficient!)
            stacked_params = info["stacked_grads"]

            # Launch async all_gather:
            # - Each rank broadcasts its updated_params chunk to all other ranks
            # - Result: every rank has all updated parameters
            gather_future = dist.all_gather_into_tensor(
                stacked_params, updated_params, async_op=True
            ).get_future()

            all_gather_futures.append(
                dict(
                    gather_future=gather_future,
                    stacked_params=stacked_params,
                    params=params,
                )
            )

        # ==================== PHASE 3: Wait for all_gather and Copy Parameters Back ====================
        # Wait for all parameter broadcasts to complete, then update local parameter tensors
        for info in all_gather_futures:
            # Wait for all_gather to complete - now stacked_params contains all updated params
            info["gather_future"].wait()
            stacked_params = info["stacked_params"]
            params = info["params"]

            # Copy gathered parameters back to original parameter tensors
            # torch._foreach_copy_ is a batched operation (single kernel, more efficient)
            # Only copy the actual parameters (not padding)
            torch._foreach_copy_(params, list(stacked_params[: len(params)].unbind(0)))
