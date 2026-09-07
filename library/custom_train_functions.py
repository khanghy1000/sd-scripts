from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import math
import torch
import argparse
import random
import re
from torch.types import Number
from typing import List, Optional, Union
from .utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def prepare_scheduler_for_custom_training(noise_scheduler, device):
    # upgrade scheduler precision for Grokking
    if hasattr(noise_scheduler, "alphas_cumprod") and noise_scheduler.alphas_cumprod is not None:
        noise_scheduler.alphas_cumprod = noise_scheduler.alphas_cumprod.to(dtype=torch.float64)
    
    if hasattr(noise_scheduler, "betas") and noise_scheduler.betas is not None:
        noise_scheduler.betas = noise_scheduler.betas.to(dtype=torch.float64)
        
    if hasattr(noise_scheduler, "alphas") and noise_scheduler.alphas is not None:
        noise_scheduler.alphas = noise_scheduler.alphas.to(dtype=torch.float64)

    if hasattr(noise_scheduler, "all_snr"):
        return

    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    alpha = sqrt_alphas_cumprod
    sigma = sqrt_one_minus_alphas_cumprod
    all_snr = (alpha / sigma) ** 2

    noise_scheduler.all_snr = all_snr.to(device)


def fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler):
    # fix beta: zero terminal SNR
    logger.info(f"fix noise scheduler betas: https://arxiv.org/abs/2305.08891")

    def enforce_zero_terminal_snr(betas):
        # Cast to float64 for grokking
        betas = betas.to(dtype=torch.float64)

        # Convert betas to alphas_bar_sqrt
        alphas = 1 - betas
        alphas_bar = alphas.cumprod(0)
        alphas_bar_sqrt = alphas_bar.sqrt()

        # Store old values.
        alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
        alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
        # Shift so last timestep is zero.
        alphas_bar_sqrt -= alphas_bar_sqrt_T
        # Scale so first timestep is back to old value.
        alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

        # Convert alphas_bar_sqrt to betas
        alphas_bar = alphas_bar_sqrt**2
        alphas = alphas_bar[1:] / alphas_bar[:-1]
        alphas = torch.cat([alphas_bar[0:1], alphas])
        betas = 1 - alphas
        return betas

    betas = noise_scheduler.betas
    betas = enforce_zero_terminal_snr(betas)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # logger.info(f"original: {noise_scheduler.betas}")
    # logger.info(f"fixed: {betas}")

    noise_scheduler.betas = betas
    noise_scheduler.alphas = alphas
    noise_scheduler.alphas_cumprod = alphas_cumprod

def apply_snr_weight(
    loss: torch.Tensor,
    timesteps: torch.IntTensor,
    noise_scheduler,
    gamma: float,
    v_prediction: bool = False,
    soft: bool = False,
):
    """Apply Min-SNR or Soft Min-SNR weighting to the loss.

    Args:
        loss: Per-element loss tensor.
        timesteps: Timesteps for each sample in the batch.
        noise_scheduler: DDPMScheduler containing the precomputed SNR values.
        gamma: Min-SNR clipping value (typically 4.0 or 5.0).
        v_prediction: Set to True if the model predicts velocity (v).
        soft: Set to True to use the smooth Soft Min-SNR transition from the paper.
    """
    # Retrieve SNR values and move to the correct device
    snr = torch.stack([noise_scheduler.all_snr[t] for t in timesteps]).to(device=loss.device)

    if soft:
        if v_prediction:
            snr_weight = (snr * gamma) / ((snr + gamma) * (snr + 1))
        else:
            snr_weight = gamma / (snr + gamma)
    else:
        min_snr_gamma = torch.minimum(snr, torch.full_like(snr, gamma))
        if v_prediction:
            snr_weight = torch.div(min_snr_gamma, snr + 1)
        else:
            snr_weight = torch.div(min_snr_gamma, snr)

    snr_weight = snr_weight.to(dtype=loss.dtype)

    # Ensure snr_weight dimensions match loss for proper broadcasting
    while snr_weight.ndim < loss.ndim:
        snr_weight = snr_weight.unsqueeze(-1)

    return loss * snr_weight


def apply_snr_weight_for_flow_matching(
    loss: torch.Tensor, 
    sigmas: torch.Tensor, 
    gamma: float, 
    soft: bool = False
) -> torch.Tensor:
    """Apply Min-SNR-γ or Soft Min-SNR-γ weighting for flow matching models.

    Computes the signal-to-noise ratio from sigma: SNR = (1 - σ)² / σ²
    and applies the velocity-prediction weight.

    Args:
        loss: Per-element loss tensor (any shape, e.g. (B,) or (B, C, H, W)).
        sigmas: Noise levels from the flow matching scheduler.
            Can be shape (B, 1, 1, 1), (B,), or broadcastable with loss.
        gamma: Min-SNR gamma value.
        soft: Set to True to use the smooth Soft Min-SNR transition from the paper.

    Returns:
        Weighted loss tensor (same shape as input loss).
    """
    # Clamp sigma away from zero to avoid division by zero at σ=0 (clean data, infinite SNR)
    sigma = sigmas.clamp(min=1e-6)

    # SNR in flow matching: (1 - σ)² / σ²
    snr = ((1.0 - sigma) / sigma) ** 2

    if soft:
        # Velocity prediction weight using Soft Min-SNR
        snr_weight = (snr * gamma) / ((snr + gamma) * (snr + 1))
    else:
        # Cap SNR at gamma
        min_snr = torch.minimum(snr, torch.full_like(snr, gamma))
        # Velocity prediction weight: min(SNR, γ) / (SNR + 1)
        snr_weight = min_snr / (snr + 1)

    snr_weight = snr_weight.to(dtype=loss.dtype, device=loss.device)

    return loss * snr_weight


class _QMCSequenceManager:
    """Manages low-discrepancy (quasi-random) sequences for timestep sampling.

    Sobol and Halton sequences are deterministic low-discrepancy sequences that
    fill the unit interval more uniformly than pseudo-random numbers, yielding
    faster convergence of Monte Carlo estimates (variance ~O((log B)^d / B^d)
    vs ~O(1/B) for iid). Unlike stratified sampling (which resets every batch),
    a QMC sequence advances across batches so that over many steps the entire
    timestep range is covered with minimal discrepancy.

    The manager keeps a global draw counter so consecutive calls produce
    *different* points (the sequence does not restart at 0 each batch). A
    scrambled Sobol engine is used for randomization, which preserves the
    low-discrepancy property while allowing unbiased error estimation.

    Multi-GPU (DDP) support
    ------------------------
    Each DDP rank is a separate Python process with its own class-level
    ``_instances`` dict. If every rank used the same ``seed`` they would all
    draw the *identical* sequence, so the combined effective batch would be the
    same ``batch_size`` points repeated ``num_processes`` times — destroying the
    low-discrepancy property across the DDP dimension. To avoid this, callers
    pass ``rank`` (the DDP process index); the manager offsets the scramble
    seed by ``rank`` so each rank draws from a *different* scrambled sequence.
    Scrambling preserves the low-discrepancy property of each individual
    sequence, so every rank still gets a space-filling set of points, and the
    combined batch across ranks has no duplicate points. (This is simpler and
    more robust than fast-forwarding a single global sequence, which would
    require the total draw count to remain a power of two for Sobol's balance
    properties.)

    Supported methods:
        "sobol":  Scrambled Sobol sequence (torch.quasirandom.SobolEngine).
        "halton": Halton sequence (scipy.stats.qmc.Halton, scrambled).
        "sobol_jittered": Sobol sequence used as within-stratum jitter; each
            batch of ``n`` draws is placed one point per equal stratum of
            [0,1] (strata assignment randomly permuted), combining per-batch
            stratified coverage with low-discrepancy jitter across batches.
        "halton_jittered": Same, with a Halton sequence as the jitter source.

    Sobol base-2 alignment
    ----------------------
    Sobol sequences only have their optimal balance (low-discrepancy)
    properties when the *total* number of generated points is a power of two.
    When a draw of ``n = 2**m`` points is requested and the running total is
    not a multiple of ``n``, the manager first draws and discards
    ``(-total) % n`` padding points so every batch is a balanced base-2 block.
    Padding is counted in ``_draw_count`` so checkpoint save/resume fast-forward
    reproduces the exact same sequence position.

    Note on performance
    -------------------
    Both Sobol (``torch.quasirandom.SobolEngine``) and Halton (``scipy.stats.qmc``)
    generate points on the CPU; the result is then moved to the target device.
    For typical batch sizes (1-32) this host->device transfer is negligible, but
    it is a per-step sync point. There is no native CUDA Sobol/Halton generator
    in torch, so this cost is unavoidable with the current engines.
    """

    _instances: dict = {}  # keyed by (method, seed) -> _QMCSequenceManager

    def __new__(cls, method: str = "sobol", seed: int = 0, rank: int = 0):
        key = (method, seed, rank)
        if key not in cls._instances:
            cls._instances[key] = super().__new__(cls)
            cls._instances[key]._initialized = False
        return cls._instances[key]

    def __init__(self, method: str = "sobol", seed: int = 0, rank: int = 0):
        if self._initialized:
            return
        self.method = method
        self.seed = seed
        self.rank = rank
        # Total number of points consumed so far (across all draws). Used for
        # checkpoint save/resume fast-forward (see state_dict/load_state_dict).
        self._draw_count = 0
        # Offset the scramble seed by rank so each DDP rank draws from a
        # different scrambled low-discrepancy sequence (no duplicate points
        # across ranks, each still space-filling).
        effective_seed = seed + rank
        # Jittered variants use the named base sequence as within-stratum
        # jitter; one point per stratum per batch (stratified-QMC hybrid).
        self.jittered = method.endswith("_jittered")
        self._engine_kind = "sobol" if method.startswith("sobol") else ("halton" if method.startswith("halton") else None)
        if self._engine_kind == "sobol":
            # Scrambled Sobol for unbiased error estimation; dimension=1 (timestep).
            self._sobol = torch.quasirandom.SobolEngine(dimension=1, scramble=True, seed=effective_seed)
        elif self._engine_kind == "halton":
            try:
                from scipy.stats import qmc as scipy_qmc
            except ImportError as e:
                raise ImportError(
                    "Halton QMC timestep sampling requires scipy. Install it with: pip install scipy"
                ) from e
            self._scipy_qmc = scipy_qmc
            self._halton = scipy_qmc.Halton(d=1, scramble=True, seed=effective_seed)
        else:
            raise ValueError(
                f"Unknown QMC method: {method!r}. Use 'sobol', 'halton', 'sobol_jittered' or 'halton_jittered'."
            )
        self._initialized = True

    @classmethod
    def clear_instances(cls):
        """Drop all cached singleton instances.

        Useful for test isolation (so one test's drawn-ahead sequence does not
        leak into the next) and for forcing a fresh sequence on demand. After
        calling this, subsequent ``_QMCSequenceManager(...)`` constructions build
        new engines from their seed.
        """
        cls._instances.clear()

    def draw(self, n: int, device: Union[str, torch.device] = "cpu") -> torch.Tensor:
        """Draw the next ``n`` points of the low-discrepancy sequence.

        Returns a tensor of shape (n,) in [0, 1] on ``device``.
        """
        if n <= 0:
            # Guard: a zero/negative draw is a no-op returning an empty tensor.
            # Avoids passing 0 to SobolEngine.draw / Halton.random (undefined).
            return torch.empty(0, device=device, dtype=torch.float32)
        if self._engine_kind == "sobol":
            # Sobol sequences have optimal low-discrepancy when the *total* number
            # of generated points is a power of two. When n is a power of two,
            # pad the running total up to a multiple of n (discarding the
            # padding points) so every batch is a balanced base-2 block drawn
            # via draw_base2; otherwise fall back to draw() (always valid, just
            # slightly less optimal).
            n_is_pow2 = n > 0 and (n & (n - 1)) == 0
            if n_is_pow2:
                # Align the running total to a multiple of n (discarding the
                # padding points) so every batch is a balanced Sobol block: any
                # 2^m consecutive points starting at a multiple of 2^m form a
                # (0,m,1)-net. Use draw_base2 when the cumulative total lands
                # on a power of two (torch's API requires this); otherwise a
                # plain aligned draw() is still balanced and always valid.
                pad = (-self._draw_count) % n
                if pad:
                    self._sobol.draw(pad)
                    self._draw_count += pad
                total_after = self._draw_count + n
                if (total_after & (total_after - 1)) == 0:
                    pts = self._sobol.draw_base2(int(math.log2(n))).squeeze(-1)
                else:
                    pts = self._sobol.draw(n).squeeze(-1)
            else:
                # SobolEngine draws on CPU; move to target device.
                pts = self._sobol.draw(n).squeeze(-1)
            pts = pts.to(dtype=torch.float32)
        else:
            # Halton via scipy (CPU), then move to device.
            pts = torch.from_numpy(self._halton.random(n).squeeze(-1)).to(dtype=torch.float32)
        self._draw_count += n
        if self.jittered:
            # Stratified-QMC hybrid: use each sequence point as the within-
            # stratum offset and assign points to a random permutation of the n
            # equal strata, guaranteeing one point per stratum per batch.
            perm = torch.randperm(n, dtype=torch.float32)
            pts = (perm + pts) / n
        return pts.to(device=device)

    def reset(self):
        """Reset the sequence to the beginning (e.g. for reproducibility)."""
        self._draw_count = 0
        effective_seed = self.seed + self.rank
        if self._engine_kind == "sobol":
            self._sobol = torch.quasirandom.SobolEngine(dimension=1, scramble=True, seed=effective_seed)
        else:
            self._halton = self._scipy_qmc.Halton(d=1, scramble=True, seed=effective_seed)

    def state_dict(self) -> dict:
        """Return a serializable snapshot of the sequence position.

        Only the draw count is needed to reconstruct the sequence position
        (the engine is deterministic given the seed and rank). The rank is
        re-derived on construction, so it is stored for traceability only.
        """
        return {"method": self.method, "seed": self.seed, "rank": self.rank, "draw_count": self._draw_count}

    def load_state_dict(self, state: dict):
        """Restore the sequence position from a ``state_dict`` snapshot.

        Rebuilds the engine from the (rank-offset) seed and fast-forwards by
        ``draw_count`` so that subsequent draws continue the sequence exactly
        where it left off. This makes QMC reproducible across checkpoint resume,
        preserving the cumulative coverage benefit.
        """
        target = int(state.get("draw_count", 0))
        # Rebuild from scratch so the fast-forward is exact.
        self._draw_count = 0
        effective_seed = self.seed + self.rank
        if self._engine_kind == "sobol":
            self._sobol = torch.quasirandom.SobolEngine(dimension=1, scramble=True, seed=effective_seed)
        else:
            self._halton = self._scipy_qmc.Halton(d=1, scramble=True, seed=effective_seed)
        # Fast-forward to the saved position.
        if target > 0:
            if self._engine_kind == "sobol":
                self._sobol.fast_forward(target)
            else:
                self._halton.random(target)
            self._draw_count = target


def compute_density_for_timestep_sampling(
    weighting_scheme: str,
    batch_size: int,
    logit_mean: float = None,
    logit_std: float = None,
    mode_scale: float = None,
    antithetic: bool = False,
    stratified: bool = False,
    qmc: str = None,
    device: Union[str, torch.device] = "cpu",
    sigmoid_scale: float = 1.0,
    qmc_seed: int = 0,
    rank: int = 0,
) -> torch.Tensor:
    """Compute the density for sampling the timesteps when doing SD3/Flux training.

    This is the single canonical implementation shared by the SD3, Flux, Lumina
    and flow-model (train_util) trainers. It supersedes the per-module copies that
    previously existed in ``sd3_train_utils``, ``flux_train_utils`` and
    ``lumina_train_util``.

    Courtesy: This was contributed by Rafie Walker in
    https://github.com/huggingface/diffusers/pull/8528.
    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.

    Variance-reduction methods (precedence order is ``antithetic`` then
    ``qmc`` then ``stratified``; antithetic and qmc compose, see below):

    * ``antithetic=True``: the base randomness is drawn as mirrored pairs
      ((z, -z) for logit_normal, (u, 1-u) for mode/uniform) before applying the
      same deterministic transform, preserving the marginal distribution while
      reducing sampling variance. Most effective at small batch sizes (4-8).
      Odd batch sizes are handled by truncating the last mirrored pair.

    * ``qmc="sobol"|"halton"``: use a low-discrepancy (quasi-random) sequence for
      the base uniform. These sequences fill [0,1] more uniformly than
      pseudo-random numbers, yielding faster convergence than iid (and often
      better than stratified at moderate batch sizes). The sequence advances
      across batches via a global counter, so over many steps the entire
      timestep range is covered with minimal discrepancy. Composes with the
      deterministic distribution transform and any shift.

    * ``stratified=True``: the unit interval is partitioned into ``batch_size``
      equal strata and one uniform is drawn inside each stratum. This guarantees
      coverage of the whole timestep range every batch and scales better than
      antithetic as batch size grows (variance ~1/B^3 vs ~1/B). Works for any
      batch size including odd. Only applies to the *base* uniform variate, so
      it composes with the deterministic distribution transform and any shift.

    Antithetic + QMC composition
    ----------------------------
    Unlike stratified, antithetic and qmc are not mutually exclusive: when
    both are set, ``batch_size // 2`` low-discrepancy points are drawn and each
    is mirrored as ``(u, 1-u)`` (for uniform/mode) or ``(z, -z)`` (for
    logit_normal/sigmoid, where ``z = logit(u)``). This "antithetic QMC"
    combines Sobol's space-filling property with pairwise variance
    cancellation, and is strictly better than either method alone in many
    settings. Odd batch sizes truncate the last mirrored pair. Only ``qmc``
    and ``stratified`` conflict (qmc wins); a warning is logged in that case.

    Args:
        weighting_scheme: One of "logit_normal", "mode", "uniform", "sigmoid".
            ("uniform" and "sigmoid" are accepted for the flow path; the SD3
            density originally only used logit_normal/mode/uniform.)
        batch_size: Number of sigmas to draw.
        logit_mean: Mean of the logit-normal base distribution.
        logit_std: Std of the logit-normal base distribution.
        mode_scale: Scale for the "mode" weighting scheme.
        antithetic: If True, draw mirrored base-variates pairs.
        stratified: If True, use stratified sampling on the base uniform.
        qmc: If set to "sobol" or "halton", use a low-discrepancy sequence for
            the base uniform.
        device: Device on which to generate the tensor. Defaults to "cpu" for
            backward compatibility with the original SD3 density, but callers on
            CUDA should pass the target device to avoid a host->device sync.
        sigmoid_scale: Scale of the normal base for "sigmoid" sampling.
        qmc_seed: Seed for the (scrambled) QMC sequence. Only used when ``qmc``
            is set.
        rank: DDP process index. When greater than 0, the QMC scramble seed is
            offset by ``rank`` so each rank draws from a different scrambled
            low-discrepancy sequence (no duplicate points across ranks, each
            still space-filling). Only used when ``qmc`` is set.

    Returns:
        Tensor of shape (batch_size,) of base variates in [0, 1] on ``device``.
    """
    # Short-circuit: a single sample cannot benefit from any variance reduction.
    if batch_size <= 1:
        antithetic = False
        stratified = False
        qmc = None

    # Resolve precedence: antithetic composes with qmc (antithetic-QMC). Only
    # qmc-vs-stratified and antithetic-vs-stratified conflict (both qmc and
    # antithetic win over stratified, since stratified replaces the base uniform
    # entirely and cannot compose with either). Warn on conflicts.
    if stratified and (qmc is not None or antithetic):
        winner = "QMC (%s)" % qmc if qmc is not None else "antithetic"
        logger.warning(
            "Both %s and stratified timestep sampling were requested; "
            "%s takes precedence and stratified is ignored." % (winner, winner)
        )
        stratified = False

    n_pairs = (batch_size + 1) // 2 if antithetic else batch_size

    # QMC produces base uniforms in [0,1]; apply the same deterministic transform.
    # When antithetic is also set, draw only n_pairs points and mirror them
    # (antithetic-QMC composition: combines low-discrepancy coverage with
    # pairwise variance cancellation).
    if qmc is not None:
        qmc_mgr = _QMCSequenceManager(method=qmc, seed=qmc_seed, rank=rank)
        u_base = qmc_mgr.draw(n_pairs, device=device)

    if weighting_scheme == "logit_normal":
        # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
        if qmc is not None:
            # Map the low-discrepancy uniform through the inverse-CDF (logit)
            # to get a low-discrepancy sample under the logit-normal distribution.
            z = torch.logit(u_base.clamp(1e-7, 1 - 1e-7))
            if antithetic:
                # Antithetic-QMC: mirror the standardized z so the pair is
                # symmetric about the mean.
                z = torch.cat([z, -z])[:batch_size]
            u = torch.nn.functional.sigmoid(logit_mean + logit_std * z)
        else:
            # Mirror the standardized z so the pair is symmetric about the mean.
            z = torch.normal(mean=0.0, std=1.0, size=(n_pairs,), device=device)
            if antithetic:
                z = torch.cat([z, -z])[:batch_size]
            u = torch.nn.functional.sigmoid(logit_mean + logit_std * z)
    elif weighting_scheme == "mode":
        if qmc is not None:
            if antithetic:
                # Antithetic-QMC: mirror the base uniform before the transform.
                u_base = torch.cat([u_base, 1.0 - u_base])[:batch_size]
            u = 1 - u_base - mode_scale * (torch.cos(math.pi * u_base / 2) ** 2 - 1 + u_base)
        else:
            u = torch.rand(size=(n_pairs,), device=device)
            if antithetic:
                u = torch.cat([u, 1.0 - u])[:batch_size]
            u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    elif weighting_scheme == "sigmoid":
        # XLabs-AI style: sigma = sigmoid(scale * z).
        if qmc is not None:
            z = torch.logit(u_base.clamp(1e-7, 1 - 1e-7))
            if antithetic:
                # Antithetic-QMC: mirror the standardized z.
                z = torch.cat([z, -z])[:batch_size]
            u = torch.sigmoid(sigmoid_scale * z)
        else:
            z = torch.normal(mean=0.0, std=1.0, size=(n_pairs,), device=device)
            if antithetic:
                z = torch.cat([z, -z])[:batch_size]
            u = torch.sigmoid(sigmoid_scale * z)
    else:
        # "uniform" (and any unknown scheme falls back to uniform).
        if qmc is not None:
            if antithetic:
                # Antithetic-QMC: mirror the base uniform.
                u_base = torch.cat([u_base, 1.0 - u_base])[:batch_size]
            u = u_base
        elif stratified:
            # One uniform per equal-width stratum: guarantees full [0,1] coverage.
            edges = torch.arange(batch_size, device=device, dtype=torch.float32)
            u = (edges + torch.rand(batch_size, device=device)) / batch_size
        else:
            u = torch.rand(size=(n_pairs,), device=device)
            if antithetic:
                u = torch.cat([u, 1.0 - u])[:batch_size]
    return u


def compute_antithetic_sigmas(
    batch_size: int,
    distribution: str,
    device: torch.device,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    sigmoid_scale: float = 1.0,
) -> torch.Tensor:
    """Sample flow-matching sigmas with antithetic pairing for variance reduction.

    .. deprecated::
        Thin backward-compatible wrapper around
        :func:`compute_density_for_timestep_sampling` with ``antithetic=True``.
        New code should call that function directly.

    The batch is filled with mirrored pairs of the *base* randomness, and the
    configured distribution transform is applied identically to both members of
    each pair. Because the base variate of a mirrored pair (z, -z) or (u, 1-u)
    has the same marginal distribution as an i.i.d. draw, the batch remains
    marginally distributed according to the configured distribution while
    cancelling a large fraction of sampling variance.

    Supported distributions:
        "uniform":      u ~ U(0,1);           pair = (u, 1-u)
        "logit_normal": sigma = sigmoid(mean + std*z), z ~ N(0,1); pair = (z, -z)
        "sigmoid":      sigma = sigmoid(scale*z), z ~ N(0,1);       pair = (z, -z)
        "mode":         u ~ U(0,1) transformed; pair = (u, 1-u)

    Any downstream *deterministic* transform of sigma (e.g. the SD3/Flux shift
    sigma' = s*sigma / (1 + (s-1)*sigma)) may be applied afterwards and still
    respects the intended final distribution.

    Args:
        batch_size: Number of sigmas to draw. Odd batch sizes are handled by
            truncating the last mirrored pair.
        distribution: One of "uniform", "logit_normal", "sigmoid", "mode".
        device: Torch device for the returned tensor.
        logit_mean: Mean of the logit-normal base distribution.
        logit_std: Std of the logit-normal base distribution.
        sigmoid_scale: Scale of the normal base for "sigmoid" sampling.

    Returns:
        Tensor of shape (batch_size,) on ``device``, float32.
    """
    # Preserve the strict validation of the original implementation: only the
    # explicitly supported distributions are accepted here (the canonical
    # density function falls back to uniform for unknown schemes, which is the
    # desired behavior for the SD3 weighting_scheme path but not for this
    # dedicated antithetic helper).
    _SUPPORTED = ("uniform", "logit_normal", "sigmoid", "mode")
    if distribution not in _SUPPORTED:
        raise ValueError(f"Unknown antithetic sigma distribution: {distribution}")
    return compute_density_for_timestep_sampling(
        weighting_scheme=distribution,
        batch_size=batch_size,
        logit_mean=logit_mean,
        logit_std=logit_std,
        mode_scale=1.29,  # SD3 default; only used for "mode"
        antithetic=True,
        device=device,
        sigmoid_scale=sigmoid_scale,
    )


def apply_flow_shift(sigmas: torch.Tensor, shift) -> torch.Tensor:
    """Apply the SD3/Flux timestep shift: sigma' = s*sigma / (1 + (s-1)*sigma).

    Args:
        sigmas: Tensor of sigmas in [0, 1].
        shift: Positive scalar or per-sample tensor of shift ratios.
    """
    return (sigmas * shift) / (1.0 + (shift - 1.0) * sigmas)


def apply_antithetic_noise_pairing(noise: torch.Tensor) -> torch.Tensor:
    """Mirror the noise tensor as antithetic pairs along the batch dimension.

    Given a noise tensor of shape (B, ...), the first ``ceil(B/2)`` entries are
    kept and the remaining entries are replaced by their negation, so sample
    ``i`` and sample ``i + ceil(B/2)`` form a mirrored pair ``(eps, -eps)``.
    This matches the pair ordering of
    :func:`compute_density_for_timestep_sampling` with ``antithetic=True``
    (base draws first, mirrored copies second), so composing both pairs the
    timestep *and* the noise of each sample with its partner.

    Because ``-eps`` has the same marginal distribution as ``eps`` for a
    symmetric Gaussian, the marginal noise distribution is unchanged; only the
    joint (within-batch) correlation structure is altered, which cancels a
    large fraction of gradient variance from the flow-matching target
    ``v = eps - x0``.

    Odd batch sizes truncate the last mirrored pair. Batch sizes <= 1 are a
    no-op (a single sample cannot benefit from pairing).

    Args:
        noise: Noise tensor of shape (B, ...).

    Returns:
        Tensor of the same shape and dtype with mirrored-pair structure.
    """
    bsz = noise.shape[0]
    if bsz <= 1:
        return noise
    n_pairs = (bsz + 1) // 2
    base = noise[:n_pairs]
    return torch.cat([base, -base], dim=0)[:bsz]


def maybe_apply_antithetic_noise_pairing(args, noise: torch.Tensor, is_train: bool = True) -> torch.Tensor:
    """Apply :func:`apply_antithetic_noise_pairing` when enabled on ``args``.

    Convenience guard for trainer call sites: pairs the noise only during
    training and only when ``--antithetic_noise_pairing`` is set. Call this
    immediately after sampling (and any OT reassignment of) the noise, so the
    paired noise is used both for the noisy input *and* the loss target.
    """
    if is_train and getattr(args, "antithetic_noise_pairing", False):
        return apply_antithetic_noise_pairing(noise)
    return noise


def apply_token_mining(
    loss: torch.Tensor,
    sigmas: Optional[torch.Tensor] = None,
    alpha: float = 1.0,
    min_weight: float = 0.25,
    max_weight: float = 4.0,
    sigma_gate: bool = True,
) -> torch.Tensor:
    """Token-level hard-example mining for per-element (spatial) losses.

    Computes a per-token difficulty map from the *detached* per-element loss,
    converts it to multiplicative weights, and reweights the loss so that hard
    spatial tokens (edges, textures) contribute more gradient than easy ones
    (flat regions). Weights are detached, so the model cannot inflate its own
    mining weights.

    Weight construction per sample:
        w_i = clamp((L_i / median(L)) ** alpha, min_weight, max_weight)
    followed by renormalization to mean 1 per sample, so the overall loss scale
    matches the plain mean reduction.

    When ``sigma_gate`` is enabled and ``sigmas`` are provided, mining strength
    is gated by g(sigma) = clip(4*sigma*(1-sigma), 0, 1): full strength at
    mid-schedule, disabled at the sigma extremes (where per-token loss
    variation is mostly irreducible noise). The gate blends weights toward
    uniform: w = 1 + g*(w-1), again renormalized to mean 1.

    Args:
        loss: Per-element loss, shape (B, C, ...) with >= 3 dims (e.g. (B,C,H,W)).
            Tensors with fewer than 3 dims are returned unchanged.
        sigmas: Optional per-sample flow-matching sigmas, shape (B,) or (B,1,...).
        alpha: Difficulty exponent. Higher values concentrate more weight on
            hard tokens.
        min_weight / max_weight: Clamp bounds for the mining weights, relative
            to uniform (1.0).
        sigma_gate: Enable the sigma-dependent strength gate.

    Returns:
        Weighted loss tensor, same shape and dtype as ``loss``.
    """
    if loss.ndim < 3:
        return loss

    with torch.no_grad():
        per_token = loss.detach().to(torch.float32).mean(dim=1)  # (B, ...) e.g. (B,H,W)
        flat = per_token.flatten(1)  # (B, N)
        med = flat.median(dim=1, keepdim=True).values.clamp(min=1e-12)
        w = (flat / med) ** alpha
        w = w.clamp(min_weight, max_weight)
        w = w / w.mean(dim=1, keepdim=True).clamp(min=1e-12)

        if sigma_gate and sigmas is not None:
            s = sigmas.detach().reshape(sigmas.shape[0], -1)[:, 0].to(torch.float32).clamp(0.0, 1.0)
            g = (4.0 * s * (1.0 - s)).clamp(0.0, 1.0).unsqueeze(1)  # (B, 1)
            w = 1.0 + g * (w - 1.0)
            w = w / w.mean(dim=1, keepdim=True).clamp(min=1e-12)

        w = w.view(per_token.shape).unsqueeze(1)  # (B, 1, ...)

    return loss * w.to(dtype=loss.dtype, device=loss.device)


def scale_v_prediction_loss_like_noise_prediction(loss: torch.Tensor, timesteps: torch.IntTensor, noise_scheduler: DDPMScheduler):
    scale = get_snr_scale(timesteps, noise_scheduler)
    loss = loss * scale
    return loss


def get_snr_scale(timesteps: torch.IntTensor, noise_scheduler: DDPMScheduler):
    snr_t = torch.stack([noise_scheduler.all_snr[t] for t in timesteps])  # batch_size
    snr_t = torch.minimum(snr_t, torch.ones_like(snr_t) * 1000)  # if timestep is 0, snr_t is inf, so limit it to 1000
    scale = snr_t / (snr_t + 1)
    # # show debug info
    # logger.info(f"timesteps: {timesteps}, snr_t: {snr_t}, scale: {scale}")
    return scale


def add_v_prediction_like_loss(loss: torch.Tensor, timesteps: torch.IntTensor, noise_scheduler: DDPMScheduler, v_pred_like_loss: torch.Tensor):
    scale = get_snr_scale(timesteps, noise_scheduler)
    # logger.info(f"add v-prediction like loss: {v_pred_like_loss}, scale: {scale}, loss: {loss}, time: {timesteps}")
    loss = loss + loss / scale * v_pred_like_loss
    return loss


def apply_debiased_estimation(loss: torch.Tensor, timesteps: torch.IntTensor, noise_scheduler: DDPMScheduler, v_prediction=False):
    snr_t = torch.stack([noise_scheduler.all_snr[t] for t in timesteps])  # batch_size
    snr_t = torch.minimum(snr_t, torch.ones_like(snr_t) * 1000)  # if timestep is 0, snr_t is inf, so limit it to 1000
    if v_prediction:
        weight = 1 / (snr_t + 1)
    else:
        weight = 1 / torch.sqrt(snr_t)
    loss = weight * loss
    return loss


# TODO train_utilと分散しているのでどちらかに寄せる


def add_custom_train_arguments(parser: argparse.ArgumentParser, support_weighted_captions: bool = True):
    parser.add_argument(
        "--min_snr_gamma",
        type=float,
        default=None,
        help="gamma for reducing the weight of high loss timesteps. Lower numbers have stronger effect. 5 is recommended by paper. / 低いタイムステップでの高いlossに対して重みを減らすためのgamma値、低いほど効果が強く、論文では5が推奨",
    )

    parser.add_argument(
        "--min_snr_gamma_soft",
        action="store_true",
        help="Controls if min_snr_gamma uses soft implementation from https://arxiv.org/abs/2401.11605.",
    )

    parser.add_argument(
        "--scale_v_pred_loss_like_noise_pred",
        action="store_true",
        help="scale v-prediction loss like noise prediction loss / v-prediction lossをnoise prediction lossと同じようにスケーリングする",
    )
    parser.add_argument(
        "--v_pred_like_loss",
        type=float,
        default=None,
        help="add v-prediction like loss multiplied by this value / v-prediction lossをこの値をかけたものをlossに加算する",
    )
    parser.add_argument(
        "--debiased_estimation_loss",
        action="store_true",
        help="debiased estimation loss / debiased estimation loss",
    )
    # Focal Frequency Loss arguments
    parser.add_argument(
        "--focal_frequency_loss",
        action="store_true",
        help="Enable focal frequency loss as an auxiliary loss in latent space. "
        "Penalizes hard-to-synthesize frequency components in the noise prediction. "
        "/ 潜在空間でfocal frequency lossを補助損失として有効にする。"
        "ノイズ予測における合成困難な周波数成分にペナルティを与える",
    )
    parser.add_argument(
        "--focal_frequency_loss_weight",
        type=float,
        default=1.0,
        help="Weight for focal frequency loss (default: 1.0) / "
        "focal frequency lossの重み（デフォルト: 1.0）",
    )
    parser.add_argument(
        "--focal_frequency_loss_alpha",
        type=float,
        default=1.0,
        help="Alpha scaling factor for the spectrum weight matrix in FFL. "
        "Controls how focused the model is on hard frequencies (default: 1.0) / "
        "FFLのスペクトル重み行列のalphaスケーリング係数。"
        "モデルが困難な周波数にどれだけ集中するかを制御する（デフォルト: 1.0）",
    )
    # Patch Topology Loss arguments
    parser.add_argument(
        "--patch_topology_loss",
        action="store_true",
        help="Enable VAE-Free Independent Patch Self-Similarity Topology Loss. "
        "Computes spatial patch affinity matrices on predicted and target representations "
        "and matches their topology across multi-scale octaves. / "
        "VAEフリーのパッチ自己類似度トポロジー損失を有効にする。"
        "予測表現とターゲット表現の空間パッチアフィニティ行列を計算し、"
        "マルチスケールオクターブ間でトポロジーを一致させる",
    )
    parser.add_argument(
        "--patch_topology_weight",
        type=float,
        default=1.0,
        help="Overall loss weight scaling factor for Patch Topology Loss (default: 1.0) / "
        "Patch Topology Lossの全体損失重みスケーリング係数（デフォルト: 1.0）",
    )
    parser.add_argument(
        "--patch_topology_tau",
        type=float,
        default=0.1,
        help="Softmax temperature scaling factor for patch affinity distributions (default: 0.1) / "
        "パッチアフィニティ分布のSoftmax温度スケーリング係数（デフォルト: 0.1）",
    )
    parser.add_argument(
        "--patch_topology_scale_levels",
        type=int,
        default=2,
        help="Number of spatial pyramid octaves for Patch Topology Loss (default: 2) / "
        "Patch Topology Lossの空間ピラミッドオクターブ数（デフォルト: 2）",
    )
    parser.add_argument(
        "--patch_topology_loss_type",
        type=str,
        default="kl",
        help="Distance metric between patch affinity distributions ('kl', 'ce', 'cosine', 'l2') "
        "(default: 'kl') / パッチアフィニティ分布間の距離指標（デフォルト: 'kl'）",
    )
    parser.add_argument(
        "--patch_topology_disable_timestep_weight",
        action="store_true",
        help="Disable timestep decay weighting (1 - t) in Patch Topology Loss. / "
        "Patch Topology Lossのタイムステップ減衰重み付け（1 - t）を無効にする",
    )
    parser.add_argument(
        "--patch_topology_chunk_size",
        type=int,
        default=512,
        help="Chunk size for spatial query patches in Patch Topology Loss to limit VRAM usage (default: 512) / "
        "Patch Topology LossのVRAM使用量を制限するための空間クエリパッチのチャンクサイズ（デフォルト: 512）",
    )
    parser.add_argument(
        "--patch_topology_start_step",
        type=int,
        default=0,
        help="Training step at which to start applying Patch Topology Loss (default: 0). "
        "Before this step, the loss is skipped entirely. / "
        "Patch Topology Lossの適用を開始するトレーニングステップ（デフォルト: 0）。"
        "このステップ以前は損失は完全にスキップされる",
    )
    parser.add_argument(
        "--patch_topology_warmup_steps",
        type=int,
        default=0,
        help="Number of steps to linearly ramp Patch Topology Loss weight from 0 to full weight "
        "after start_step (default: 0 = no warmup). / "
        "start_step後にPatch Topology Lossの重みを0から目標重みまで線形に増加させるステップ数"
        "（デフォルト: 0 = ウォームアップなし）",
    )
    parser.add_argument(
        "--patch_topology_dynamic_weighting",
        type=str,
        default="none",
        choices=["none", "dwa", "gradnorm"],
        help="Dynamic multi-loss weighting strategy for Patch Topology Loss relative to the base loss. "
        "'none': static patch_topology_weight; 'dwa': Dynamic Weight Averaging by recent loss decrease rates; "
        "'gradnorm': direct GradNorm balancing gradient norms on trainable network parameters "
        "(default: 'none') / "
        "Patch Topology Lossの動的マルチ損失重み付け戦略。"
        "'none': 静的な重み、'dwa': 最近の損失減少率による動的重み平均、"
        "'gradnorm': 勾配ノルムに基づくGradNormバランシング（デフォルト: 'none'）",
    )
    parser.add_argument(
        "--patch_topology_dwa_temperature",
        type=float,
        default=2.0,
        help="Temperature T for DWA dynamic weighting; higher values produce smoother weights (default: 2.0) / "
        "DWA動的重み付けの温度T。値が大きいほど重みが滑らかになる（デフォルト: 2.0）",
    )
    parser.add_argument(
        "--patch_topology_gradnorm_alpha",
        type=float,
        default=1.5,
        help="Alpha exponent controlling relative training-rate strength for GradNorm weighting (default: 1.5) / "
        "GradNorm重み付けの相対学習率の強さを制御するalpha指数（デフォルト: 1.5）",
    )
    parser.add_argument(
        "--patch_topology_dynamic_max_weight",
        type=float,
        default=10.0,
        help="Maximum clamp for dynamically-computed Patch Topology Loss weights (default: 10.0) / "
        "動的に計算されたPatch Topology Loss重みの最大クランプ値（デフォルト: 10.0）",
    )

    parser.add_argument(
        "--token_mining",
        action="store_true",
        help="Enable token-level hard-example mining on the spatial loss. Reweights latent tokens "
        "by detached per-token difficulty (median-normalized, clamped, renormalized), so hard "
        "spatial regions contribute more gradient. Best suited to flow-matching DiT training. / "
        "空間損失にトークンレベルのハードマイニングを有効にする",
    )
    parser.add_argument(
        "--token_mining_alpha",
        type=float,
        default=1.0,
        help="Difficulty exponent for token mining weights (default: 1.0). Higher concentrates more "
        "weight on hard tokens. / トークンマイニング重みの難易度指数（デフォルト: 1.0）",
    )
    parser.add_argument(
        "--token_mining_min_weight",
        type=float,
        default=0.25,
        help="Minimum mining weight relative to uniform (default: 0.25) / 一様重みに対する最小マイニング重み",
    )
    parser.add_argument(
        "--token_mining_max_weight",
        type=float,
        default=4.0,
        help="Maximum mining weight relative to uniform (default: 4.0) / 一様重みに対する最大マイニング重み",
    )
    parser.add_argument(
        "--token_mining_no_sigma_gate",
        action="store_true",
        help="Disable the sigma-dependent gate (4*sigma*(1-sigma)) that reduces mining strength at "
        "timestep extremes. / タイムステップ両端でマイニング強度を下げるシグマゲートを無効にする",
    )
    parser.add_argument(
        "--antithetic_timestep_sampling",
        action="store_true",
        help="Enable antithetic sigma sampling for flow-matching trainers: the batch is filled with "
        "mirrored pairs of the base randomness ((u, 1-u) for uniform, (z, -z) for normal-based "
        "distributions) before applying the configured distribution transform (logit_normal/uniform/"
        "sigmoid/mode) and any shift. Preserves the marginal timestep distribution while reducing "
        "sampling variance; most effective at small batch sizes (4-8). Note: with gradient "
        "accumulation or multi-GPU (DDP) the pairing may be split across micro-batches/ranks, "
        "reducing the variance-reduction benefit. / "
        "フローマッチングで対称（アンチセティック）なタイムステップサンプリングを有効にする",
    )
    parser.add_argument(
        "--stratified_timestep_sampling",
        action="store_true",
        help="Enable stratified sigma sampling for flow-matching trainers: the unit interval is "
        "partitioned into batch_size equal strata and one uniform is drawn inside each, guaranteeing "
        "full coverage of the timestep range every batch. Scales better than antithetic as batch "
        "size grows (variance ~1/B^3 vs ~1/B) and works for any batch size including odd. Only "
        "applies to the base uniform variate, so it composes with the distribution transform and "
        "shift. If both this and --qmc_timestep_sampling are set, qmc takes precedence and "
        "stratified is ignored. / "
        "フローマッチングで層化（ストラティファイド）タイムステップサンプリングを有効にする",
    )
    parser.add_argument(
        "--qmc_timestep_sampling",
        type=str,
        default=None,
        choices=["sobol", "halton", "sobol_jittered", "halton_jittered"],
        help="Enable quasi-Monte Carlo (low-discrepancy) sigma sampling for flow-matching "
        "trainers: a Sobol or Halton sequence is used for the base uniform instead of "
        "pseudo-random numbers. These sequences fill [0,1] more uniformly, yielding faster "
        "convergence than iid (and often better than stratified at moderate batch sizes). "
        "The sequence advances across batches via a global counter, so over many steps the "
        "entire timestep range is covered with minimal discrepancy. Composes with the "
        "distribution transform and shift. Antithetic and qmc compose (antithetic-QMC: "
        "batch_size//2 low-discrepancy points are drawn and mirrored); only qmc and "
        "stratified conflict (qmc wins, a warning is logged). DDP-safe: each rank "
        "offsets its scramble seed by its process index so ranks draw from different "
        "scrambled sequences (no duplicate points). The sequence position is "
        "saved/restored across checkpoint resume. 'halton' requires scipy. The "
        "'*_jittered' variants use the sequence as within-stratum jitter: one "
        "point per equal stratum of [0,1] per batch (randomly permuted strata), "
        "combining per-batch stratified coverage with low-discrepancy jitter. "
        "Sobol draws are base-2 aligned: power-of-two batch sizes always produce "
        "balanced low-discrepancy blocks. / "
        "フローマッチングで準モンテカルロ（低差異）タイムステップサンプリングを有効にする",
    )
    parser.add_argument(
        "--qmc_seed",
        type=int,
        default=0,
        help="Seed for the scrambled QMC sequence (default: 0). Only used when "
        "--qmc_timestep_sampling is set. / QMCシーケンスのシード（デフォルト: 0）",
    )
    parser.add_argument(
        "--antithetic_noise_pairing",
        action="store_true",
        help="Mirror the Gaussian noise as antithetic pairs (eps, -eps) along the batch dimension "
        "for flow-matching trainers, using the same pair ordering as --antithetic_timestep_sampling "
        "(sample i pairs with sample i + ceil(B/2)). Since -eps has the same marginal distribution "
        "as eps, this preserves the marginal while cancelling gradient variance from the "
        "flow-matching target v = eps - x0. Most effective when combined with "
        "--antithetic_timestep_sampling so each mirrored timestep pair also uses mirrored noise. "
        "Odd batch sizes truncate the last pair; batch size 1 is a no-op. / "
        "ガウシアンノイズを対称ペア（eps, -eps）としてミラーリングし、勾配分散を低減する",
    )

    if support_weighted_captions:
        parser.add_argument(
            "--weighted_captions",
            action="store_true",
            default=False,
            help="Enable weighted captions in the standard style (token:1.3). No commas inside parens, or shuffle/dropout may break the decoder. / 「[token]」、「(token)」「(token:1.3)」のような重み付きキャプションを有効にする。カンマを括弧内に入れるとシャッフルやdropoutで重みづけがおかしくなるので注意",
        )


re_attention = re.compile(
    r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:([+-]?[.\d]+)\)|
\)|
]|
[^\\()\[\]:]+|
:
""",
    re.X,
)


def parse_prompt_attention(text):
    """
    Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
    Accepted tokens are:
      (abc) - increases attention to abc by a multiplier of 1.1
      (abc:3.12) - increases attention to abc by a multiplier of 3.12
      [abc] - decreases attention to abc by a multiplier of 1.1
      \( - literal character '('
      \[ - literal character '['
      \) - literal character ')'
      \] - literal character ']'
      \\ - literal character '\'
      anything else - just text
    >>> parse_prompt_attention('normal text')
    [['normal text', 1.0]]
    >>> parse_prompt_attention('an (important) word')
    [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
    >>> parse_prompt_attention('(unbalanced')
    [['unbalanced', 1.1]]
    >>> parse_prompt_attention('\(literal\]')
    [['(literal]', 1.0]]
    >>> parse_prompt_attention('(unnecessary)(parens)')
    [['unnecessaryparens', 1.1]]
    >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
    [['a ', 1.0],
     ['house', 1.5730000000000004],
     [' ', 1.1],
     ['on', 1.0],
     [' a ', 1.1],
     ['hill', 0.55],
     [', sun, ', 1.1],
     ['sky', 1.4641000000000006],
     ['.', 1.1]]
    """

    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith("\\"):
            res.append([text[1:], 1.0])
        elif text == "(":
            round_brackets.append(len(res))
        elif text == "[":
            square_brackets.append(len(res))
        elif weight is not None and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ")" and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == "]" and len(square_brackets) > 0:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            res.append([text, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    # merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res


def get_prompts_with_weights(tokenizer, prompt: List[str], max_length: int):
    r"""
    Tokenize a list of prompts and return its tokens with weights of each token.

    No padding, starting or ending token is included.
    """
    tokens = []
    weights = []
    truncated = False
    for text in prompt:
        texts_and_weights = parse_prompt_attention(text)
        text_token = []
        text_weight = []
        for word, weight in texts_and_weights:
            # tokenize and discard the starting and the ending token
            token = tokenizer(word).input_ids[1:-1]
            text_token += token
            # copy the weight by length of token
            text_weight += [weight] * len(token)
            # stop if the text is too long (longer than truncation limit)
            if len(text_token) > max_length:
                truncated = True
                break
        # truncate
        if len(text_token) > max_length:
            truncated = True
            text_token = text_token[:max_length]
            text_weight = text_weight[:max_length]
        tokens.append(text_token)
        weights.append(text_weight)
    if truncated:
        logger.warning("Prompt was truncated. Try to shorten the prompt or increase max_embeddings_multiples")
    return tokens, weights


def pad_tokens_and_weights(tokens, weights, max_length, bos, eos, no_boseos_middle=True, chunk_length=77):
    r"""
    Pad the tokens (with starting and ending tokens) and weights (with 1.0) to max_length.
    """
    max_embeddings_multiples = (max_length - 2) // (chunk_length - 2)
    weights_length = max_length if no_boseos_middle else max_embeddings_multiples * chunk_length
    for i in range(len(tokens)):
        tokens[i] = [bos] + tokens[i] + [eos] * (max_length - 1 - len(tokens[i]))
        if no_boseos_middle:
            weights[i] = [1.0] + weights[i] + [1.0] * (max_length - 1 - len(weights[i]))
        else:
            w = []
            if len(weights[i]) == 0:
                w = [1.0] * weights_length
            else:
                for j in range(max_embeddings_multiples):
                    w.append(1.0)  # weight for starting token in this chunk
                    w += weights[i][j * (chunk_length - 2) : min(len(weights[i]), (j + 1) * (chunk_length - 2))]
                    w.append(1.0)  # weight for ending token in this chunk
                w += [1.0] * (weights_length - len(w))
            weights[i] = w[:]

    return tokens, weights


def get_unweighted_text_embeddings(
    tokenizer,
    text_encoder,
    text_input: torch.Tensor,
    chunk_length: int,
    clip_skip: int,
    eos: int,
    pad: int,
    no_boseos_middle: Optional[bool] = True,
):
    """
    When the length of tokens is a multiple of the capacity of the text encoder,
    it should be split into chunks and sent to the text encoder individually.
    """
    max_embeddings_multiples = (text_input.shape[1] - 2) // (chunk_length - 2)
    if max_embeddings_multiples > 1:
        text_embeddings = []
        for i in range(max_embeddings_multiples):
            # extract the i-th chunk
            text_input_chunk = text_input[:, i * (chunk_length - 2) : (i + 1) * (chunk_length - 2) + 2].clone()

            # cover the head and the tail by the starting and the ending tokens
            text_input_chunk[:, 0] = text_input[0, 0]
            if pad == eos:  # v1
                text_input_chunk[:, -1] = text_input[0, -1]
            else:  # v2
                for j in range(len(text_input_chunk)):
                    if text_input_chunk[j, -1] != eos and text_input_chunk[j, -1] != pad:  # 最後に普通の文字がある
                        text_input_chunk[j, -1] = eos
                    if text_input_chunk[j, 1] == pad:  # BOSだけであとはPAD
                        text_input_chunk[j, 1] = eos

            if clip_skip is None or clip_skip == 1:
                text_embedding = text_encoder(text_input_chunk)[0]
            else:
                enc_out = text_encoder(text_input_chunk, output_hidden_states=True, return_dict=True)
                text_embedding = enc_out["hidden_states"][-clip_skip]
                text_embedding = text_encoder.text_model.final_layer_norm(text_embedding)

            if no_boseos_middle:
                if i == 0:
                    # discard the ending token
                    text_embedding = text_embedding[:, :-1]
                elif i == max_embeddings_multiples - 1:
                    # discard the starting token
                    text_embedding = text_embedding[:, 1:]
                else:
                    # discard both starting and ending tokens
                    text_embedding = text_embedding[:, 1:-1]

            text_embeddings.append(text_embedding)
        text_embeddings = torch.concat(text_embeddings, axis=1)
    else:
        if clip_skip is None or clip_skip == 1:
            text_embeddings = text_encoder(text_input)[0]
        else:
            enc_out = text_encoder(text_input, output_hidden_states=True, return_dict=True)
            text_embeddings = enc_out["hidden_states"][-clip_skip]
            text_embeddings = text_encoder.text_model.final_layer_norm(text_embeddings)
    return text_embeddings


def get_weighted_text_embeddings(
    tokenizer,
    text_encoder,
    prompt: Union[str, List[str]],
    device,
    max_embeddings_multiples: Optional[int] = 3,
    no_boseos_middle: Optional[bool] = False,
    clip_skip=None,
):
    r"""
    Prompts can be assigned with local weights using brackets. For example,
    prompt 'A (very beautiful) masterpiece' highlights the words 'very beautiful',
    and the embedding tokens corresponding to the words get multiplied by a constant, 1.1.

    Also, to regularize of the embedding, the weighted embedding would be scaled to preserve the original mean.

    Args:
        prompt (`str` or `List[str]`):
            The prompt or prompts to guide the image generation.
        max_embeddings_multiples (`int`, *optional*, defaults to `3`):
            The max multiple length of prompt embeddings compared to the max output length of text encoder.
        no_boseos_middle (`bool`, *optional*, defaults to `False`):
            If the length of text token is multiples of the capacity of text encoder, whether reserve the starting and
            ending token in each of the chunk in the middle.
        skip_parsing (`bool`, *optional*, defaults to `False`):
            Skip the parsing of brackets.
        skip_weighting (`bool`, *optional*, defaults to `False`):
            Skip the weighting. When the parsing is skipped, it is forced True.
    """
    max_length = (tokenizer.model_max_length - 2) * max_embeddings_multiples + 2
    if isinstance(prompt, str):
        prompt = [prompt]

    prompt_tokens, prompt_weights = get_prompts_with_weights(tokenizer, prompt, max_length - 2)

    # round up the longest length of tokens to a multiple of (model_max_length - 2)
    max_length = max([len(token) for token in prompt_tokens])

    max_embeddings_multiples = min(
        max_embeddings_multiples,
        (max_length - 1) // (tokenizer.model_max_length - 2) + 1,
    )
    max_embeddings_multiples = max(1, max_embeddings_multiples)
    max_length = (tokenizer.model_max_length - 2) * max_embeddings_multiples + 2

    # pad the length of tokens and weights
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    prompt_tokens, prompt_weights = pad_tokens_and_weights(
        prompt_tokens,
        prompt_weights,
        max_length,
        bos,
        eos,
        no_boseos_middle=no_boseos_middle,
        chunk_length=tokenizer.model_max_length,
    )
    prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=device)

    # get the embeddings
    text_embeddings = get_unweighted_text_embeddings(
        tokenizer,
        text_encoder,
        prompt_tokens,
        tokenizer.model_max_length,
        clip_skip,
        eos,
        pad,
        no_boseos_middle=no_boseos_middle,
    )
    prompt_weights = torch.tensor(prompt_weights, dtype=text_embeddings.dtype, device=device)

    # assign weights to the prompts and normalize in the sense of mean
    previous_mean = text_embeddings.float().mean(axis=[-2, -1]).to(text_embeddings.dtype)
    text_embeddings = text_embeddings * prompt_weights.unsqueeze(-1)
    current_mean = text_embeddings.float().mean(axis=[-2, -1]).to(text_embeddings.dtype)
    text_embeddings = text_embeddings * (previous_mean / current_mean).unsqueeze(-1).unsqueeze(-1)

    return text_embeddings


# https://wandb.ai/johnowhitaker/multires_noise/reports/Multi-Resolution-Noise-for-Diffusion-Model-Training--VmlldzozNjYyOTU2
def pyramid_noise_like(noise, device, iterations=6, discount=0.4) -> torch.FloatTensor:
    b, c, w, h = noise.shape  # EDIT: w and h get over-written, rename for a different variant!
    u = torch.nn.Upsample(size=(w, h), mode="bilinear").to(device)
    for i in range(iterations):
        r = random.random() * 2 + 2  # Rather than always going 2x,
        wn, hn = max(1, int(w / (r**i))), max(1, int(h / (r**i)))
        noise += u(torch.randn(b, c, wn, hn).to(device)) * discount**i
        if wn == 1 or hn == 1:
            break  # Lowest resolution is 1x1
    return noise / noise.std()  # Scaled back to roughly unit variance


# https://www.crosslabs.org//blog/diffusion-with-offset-noise
def apply_noise_offset(latents, noise, noise_offset, adaptive_noise_scale) -> torch.FloatTensor:
    if noise_offset is None:
        return noise
    if adaptive_noise_scale is not None:
        # latent shape: (batch_size, channels, height, width)
        # abs mean value for each channel
        latent_mean = torch.abs(latents.mean(dim=(2, 3), keepdim=True))

        # multiply adaptive noise scale to the mean value and add it to the noise offset
        noise_offset = noise_offset + adaptive_noise_scale * latent_mean
        noise_offset = torch.clamp(noise_offset, 0.0, None)  # in case of adaptive noise scale is negative

    noise = noise + noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1), device=latents.device)
    return noise


def apply_masked_loss(loss, batch) -> torch.FloatTensor:
    if "conditioning_images" in batch:
        # conditioning image is -1 to 1. we need to convert it to 0 to 1
        mask_image = batch["conditioning_images"].to(dtype=loss.dtype)[:, 0].unsqueeze(1)  # use R channel
        mask_image = mask_image / 2 + 0.5
        # print(f"conditioning_image: {mask_image.shape}")
    elif "alpha_masks" in batch and batch["alpha_masks"] is not None:
        # alpha mask is 0 to 1
        mask_image = batch["alpha_masks"].to(dtype=loss.dtype).unsqueeze(1) # add channel dimension
        # print(f"mask_image: {mask_image.shape}, {mask_image.mean()}")
    else:
        return loss

    # resize to the same size as the loss
    mask_image = torch.nn.functional.interpolate(mask_image, size=loss.shape[2:], mode="area")
    loss = loss * mask_image
    return loss


"""
##########################################
# Perlin Noise
def rand_perlin_2d(device, shape, res, fade=lambda t: 6 * t**5 - 15 * t**4 + 10 * t**3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    grid = (
        torch.stack(
            torch.meshgrid(torch.arange(0, res[0], delta[0], device=device), torch.arange(0, res[1], delta[1], device=device)),
            dim=-1,
        )
        % 1
    )
    angles = 2 * torch.pi * torch.rand(res[0] + 1, res[1] + 1, device=device)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    tile_grads = (
        lambda slice1, slice2: gradients[slice1[0] : slice1[1], slice2[0] : slice2[1]]
        .repeat_interleave(d[0], 0)
        .repeat_interleave(d[1], 1)
    )
    dot = lambda grad, shift: (
        torch.stack((grid[: shape[0], : shape[1], 0] + shift[0], grid[: shape[0], : shape[1], 1] + shift[1]), dim=-1)
        * grad[: shape[0], : shape[1]]
    ).sum(dim=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[: shape[0], : shape[1]])
    return 1.414 * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1])


def rand_perlin_2d_octaves(device, shape, res, octaves=1, persistence=0.5):
    noise = torch.zeros(shape, device=device)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * rand_perlin_2d(device, shape, (frequency * res[0], frequency * res[1]))
        frequency *= 2
        amplitude *= persistence
    return noise


def perlin_noise(noise, device, octaves):
    _, c, w, h = noise.shape
    perlin = lambda: rand_perlin_2d_octaves(device, (w, h), (4, 4), octaves)
    noise_perlin = []
    for _ in range(c):
        noise_perlin.append(perlin())
    noise_perlin = torch.stack(noise_perlin).unsqueeze(0)   # (1, c, w, h)
    noise += noise_perlin # broadcast for each batch
    return noise / noise.std()  # Scaled back to roughly unit variance
"""
