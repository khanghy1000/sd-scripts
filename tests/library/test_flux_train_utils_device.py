import argparse
import pytest
import torch
from library.flux_train_utils import get_noisy_model_input_and_timesteps
from library.sd3_train_utils import FlowMatchEulerDiscreteScheduler


@pytest.fixture
def cuda_device():
    assert torch.cuda.is_available(), "CUDA must be available for testing"
    return torch.device("cuda")


@pytest.fixture
def flow_scheduler():
    # FlowMatchEulerDiscreteScheduler keeps self.timesteps on CPU by default
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
    assert scheduler.timesteps.device.type == "cpu"
    return scheduler


@pytest.mark.parametrize(
    "timestep_sampling",
    ["sigma", "uniform", "sigmoid", "shift", "flux_shift", "hump"],
)
def test_get_noisy_model_input_and_timesteps_cuda(cuda_device, flow_scheduler, timestep_sampling):
    args = argparse.Namespace(
        timestep_sampling=timestep_sampling,
        weighting_scheme="uniform",
        logit_mean=0.0,
        logit_std=1.0,
        mode_scale=1.0,
        sigmoid_scale=1.0,
        discrete_flow_shift=3.0,
        hump_center=0.5,
        ip_noise_gamma=None,
        ip_noise_gamma_random_strength=False,
    )

    bsz = 4
    latents = torch.randn(bsz, 16, 32, 32, device=cuda_device, dtype=torch.float32)
    noise = torch.randn(bsz, 16, 32, 32, device=cuda_device, dtype=torch.float32)

    noisy_input, timesteps, sigmas = get_noisy_model_input_and_timesteps(
        args=args,
        noise_scheduler=flow_scheduler,
        latents=latents,
        noise=noise,
        device=cuda_device,
        dtype=torch.float32,
    )

    assert noisy_input.device.type == "cuda"
    assert timesteps.device.type == "cuda"
    assert sigmas.device.type == "cuda"
    assert noisy_input.shape == latents.shape
    assert timesteps.shape == (bsz,)
    assert sigmas.shape == (bsz, 1, 1, 1)
    assert not torch.isnan(noisy_input).any()
    assert not torch.isnan(timesteps).any()
    assert not torch.isnan(sigmas).any()


@pytest.mark.parametrize(
    "variance_reduction",
    ["antithetic", "stratified", "qmc"],
)
def test_get_noisy_model_input_and_timesteps_cuda_variance_reduction(cuda_device, flow_scheduler, variance_reduction):
    args = argparse.Namespace(
        timestep_sampling="sigma",
        weighting_scheme="logit_normal",
        logit_mean=0.0,
        logit_std=1.0,
        mode_scale=1.29,
        sigmoid_scale=1.0,
        discrete_flow_shift=3.0,
        hump_center=0.5,
        antithetic_timestep_sampling=(variance_reduction == "antithetic"),
        stratified_timestep_sampling=(variance_reduction == "stratified"),
        qmc_timestep_sampling="sobol" if variance_reduction == "qmc" else None,
        qmc_seed=42,
        ip_noise_gamma=None,
        ip_noise_gamma_random_strength=False,
    )

    bsz = 4
    latents = torch.randn(bsz, 16, 32, 32, device=cuda_device, dtype=torch.float32)
    noise = torch.randn(bsz, 16, 32, 32, device=cuda_device, dtype=torch.float32)

    noisy_input, timesteps, sigmas = get_noisy_model_input_and_timesteps(
        args=args,
        noise_scheduler=flow_scheduler,
        latents=latents,
        noise=noise,
        device=cuda_device,
        dtype=torch.float32,
    )

    assert noisy_input.device.type == "cuda"
    assert timesteps.device.type == "cuda"
    assert sigmas.device.type == "cuda"
    assert noisy_input.shape == latents.shape
    assert timesteps.shape == (bsz,)
    assert sigmas.shape == (bsz, 1, 1, 1)
