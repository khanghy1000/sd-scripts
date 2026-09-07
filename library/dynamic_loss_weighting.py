"""
Dynamic loss weighting strategies for multi-objective (base + auxiliary) training.

Implements configurable weighting between a primary objective (e.g. diffusion
noise-prediction loss) and a single auxiliary objective (e.g. PatchTopologyLoss):

- ``none``:     static passthrough of the user-configured weight.
- ``dwa``:      Dynamic Weight Averaging (Liu et al., CVPR 2019). Weights each
                loss by the rate of its recent relative decrease, requiring no
                extra backward passes.
- ``gradnorm``: Simplified (direct) GradNorm (Chen et al., ICML 2018). Instead of
                trainable log-weights with an auxiliary optimizer, the auxiliary
                weight is solved for directly each step so that its weighted
                gradient norm on shared parameters matches the GradNorm target
                ``G_avg * r_aux**alpha``. Requires gradient access to shared
                parameters (e.g. LoRA weights) via ``torch.autograd.grad``.

In all modes the base loss keeps a fixed weight of 1.0 and only the auxiliary
loss weight is modulated, keeping the primary objective's scale untouched.
"""

import math
from collections import deque
from typing import Optional, Sequence

import torch


class DynamicLossWeighter:
    """
    Computes the effective weight for an auxiliary loss relative to a base loss.

    The base loss always has implicit weight 1.0; the returned value scales the
    auxiliary loss only.
    """

    VALID_MODES = ("none", "dwa", "gradnorm")

    def __init__(
        self,
        mode: str = "none",
        user_weight: float = 1.0,
        dwa_temperature: float = 2.0,
        gradnorm_alpha: float = 1.5,
        max_weight: float = 10.0,
        min_weight: float = 0.0,
        history_size: int = 2,
    ):
        mode = mode.lower()
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unknown dynamic weighting mode '{mode}'. Valid modes: {self.VALID_MODES}.")
        self.mode = mode
        self.user_weight = float(user_weight)
        self.dwa_temperature = max(float(dwa_temperature), 1e-6)
        self.gradnorm_alpha = float(gradnorm_alpha)
        self.max_weight = float(max_weight)
        self.min_weight = float(min_weight)
        # History of (base_loss_value, aux_loss_value) as Python floats.
        self._history: deque = deque(maxlen=max(2, history_size))
        # Initial loss values for GradNorm relative training rates.
        self._initial_base: Optional[float] = None
        self._initial_aux: Optional[float] = None

    # ------------------------------------------------------------------ utils
    def _clamp(self, w: float) -> float:
        return min(max(w, self.min_weight), self.max_weight)

    def _record(self, base_value: float, aux_value: float) -> None:
        if self._initial_base is None:
            self._initial_base = max(base_value, 1e-12)
        if self._initial_aux is None:
            self._initial_aux = max(aux_value, 1e-12)
        self._history.append((base_value, aux_value))

    # ------------------------------------------------------------------- DWA
    def _dwa_weight(self) -> float:
        """
        Dynamic Weight Averaging. With K=2 losses:
            r_i(t-1) = L_i(t-1) / L_i(t-2)
            w_i      = K * exp(r_i / T) / sum_j exp(r_j / T)
        The aux loss is scaled by user_weight * (w_aux / w_base) so the base
        loss keeps weight 1.
        """
        if len(self._history) < 2:
            return self.user_weight

        (b_prev, a_prev), (b_prev2, a_prev2) = self._history[-1], self._history[-2]
        eps = 1e-12
        r_base = b_prev / max(b_prev2, eps)
        r_aux = a_prev / max(a_prev2, eps)

        t = self.dwa_temperature
        # Clamp ratios to avoid overflow in exp for pathological spikes.
        r_base = min(max(r_base, 1e-4), 1e4)
        r_aux = min(max(r_aux, 1e-4), 1e4)

        e_base = math.exp(r_base / t)
        e_aux = math.exp(r_aux / t)
        # w_aux / w_base cancels the K factor and the denominator.
        ratio = e_aux / e_base
        return self._clamp(self.user_weight * ratio)

    # -------------------------------------------------------------- GradNorm
    @staticmethod
    def _grad_norm(loss: torch.Tensor, params: Sequence[torch.Tensor]) -> float:
        """L2 norm of d(loss)/d(params); 0.0 if grads cannot be computed."""
        try:
            grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        except RuntimeError:
            return 0.0
        total = 0.0
        for g in grads:
            if g is not None:
                total += float(g.detach().norm().item()) ** 2
        return math.sqrt(total)

    def _gradnorm_weight(
        self,
        base_loss: torch.Tensor,
        aux_loss: torch.Tensor,
        shared_params: Sequence[torch.Tensor],
    ) -> float:
        """
        Direct (non-learned) GradNorm update. GradNorm chooses weights so that
            w_i * G_i ~= G_avg * r_i,  r_i = (L_i / L_i(0))^alpha / mean_j(...),
        where G_i is the gradient norm of weighted loss i on shared params.
        With w_base fixed to 1:
            w_aux = G_avg_weighted * r_aux_norm / G_aux_raw
        """
        if not shared_params:
            return self.user_weight

        g_base = self._grad_norm(base_loss, shared_params)
        g_aux = self._grad_norm(aux_loss, shared_params)
        eps = 1e-12
        if g_aux < eps:
            return self.user_weight

        g_avg = 0.5 * (g_base + g_aux)

        base_val = max(float(base_loss.detach().item()), eps)
        aux_val = max(float(aux_loss.detach().item()), eps)
        rate_base = (base_val / max(self._initial_base, eps)) ** self.gradnorm_alpha
        rate_aux = (aux_val / max(self._initial_aux, eps)) ** self.gradnorm_alpha
        inv_rate_mean = 0.5 * (rate_base + rate_aux)
        if inv_rate_mean < eps:
            return self.user_weight
        r_aux = rate_aux / inv_rate_mean

        w = g_avg * r_aux / g_aux
        return self._clamp(w)

    # ------------------------------------------------------------------- API
    def compute_weight(
        self,
        base_loss: torch.Tensor,
        aux_loss: torch.Tensor,
        shared_params: Optional[Sequence[torch.Tensor]] = None,
    ) -> float:
        """
        Returns the effective weight to multiply the auxiliary loss by.

        Args:
            base_loss: scalar base loss tensor (graph required for gradnorm).
            aux_loss: scalar auxiliary loss tensor (graph required for gradnorm).
            shared_params: parameters used for gradient-norm computation
                (gradnorm mode only, e.g. trainable LoRA weights).
        """
        base_value = float(base_loss.detach().item())
        aux_value = float(aux_loss.detach().item())
        self._record(base_value, aux_value)

        if self.mode == "none":
            return self.user_weight
        elif self.mode == "dwa":
            # DWA uses the history including this step's values.
            return self._dwa_weight()
        else:  # gradnorm
            return self._gradnorm_weight(base_loss, aux_loss, shared_params or [])

    def state_dict(self) -> dict:
        return {
            "mode": self.mode,
            "user_weight": self.user_weight,
            "history": list(self._history),
            "initial_base": self._initial_base,
            "initial_aux": self._initial_aux,
        }

    def load_state_dict(self, state: dict) -> None:
        self._history = deque(state.get("history", []), maxlen=self._history.maxlen)
        self._initial_base = state.get("initial_base")
        self._initial_aux = state.get("initial_aux")


def build_weighter_from_args(args, user_weight: float) -> Optional[DynamicLossWeighter]:
    """Constructs a DynamicLossWeighter from parsed training args, or None if disabled."""
    mode = getattr(args, "patch_topology_dynamic_weighting", "none")
    if mode is None or mode == "none":
        return None
    return DynamicLossWeighter(
        mode=mode,
        user_weight=user_weight,
        dwa_temperature=float(getattr(args, "patch_topology_dwa_temperature", 2.0)),
        gradnorm_alpha=float(getattr(args, "patch_topology_gradnorm_alpha", 1.5)),
        max_weight=float(getattr(args, "patch_topology_dynamic_max_weight", 10.0)),
    )
