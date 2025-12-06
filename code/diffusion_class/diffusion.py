from dct_util import DCTGaussianBlur
import numpy as np
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
from einops import rearrange # type: ignore
from scipy.ndimage import zoom

from .diffusion_util import (
    mean_flat,
    ModelMeanType,
    ModelVarType,
    LossType,
    extract_into_tensor,
)

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class GaussianDiffusion(nn.Module):
    """
    Utilities for training and sampling diffusion models.

    :param betas: a 1-D numpy array of betas for each diffusion timestep, starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        gaussian_filter_std=0.0,
        img_size=None,  # maybe a tuple
        rescale_timesteps=False,
        mollifier_type="dct",  # recommend dct
        clip_min=None,
        clip_max=None,
    ):
        super().__init__()
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps
        self.base_img_size = img_size

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas = alphas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)
        assert self.alphas_cumprod_next.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

        if gaussian_filter_std == 0.0:
            self.mollifier = nn.Identity()
        elif mollifier_type == "dct":
            self.mollifier = DCTGaussianBlur(img_size, gaussian_filter_std)
        else:
            raise ValueError("mollifier_type should be dct")

        self.clip_min = clip_min
        self.clip_max = clip_max

    def compute_alpha_prod(self, idx_low, idx_high):
        """
        Compute the product of alphas from idx_low to idx_high.

        :param idx_low: The starting index of the range.
        :param idx_high: The ending index of the range.
        :return: The product of alphas in the range [idx_low, idx_high - 1].
        """
        # Ensure valid indices
        assert 0 <= idx_low < idx_high <= self.num_timesteps - 1, "Invalid indices"

        # Calculate the product of alphas in the specified range
        alpha_prod = np.prod(self.alphas[idx_low:idx_high])

        return alpha_prod

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        But note in particular that x_t returned here has been mollified
        """
        if noise is None:
            img_size = (x_start.size(-2), x_start.size(-1))

            noise_mul = min(
                img_size.shape[0] / self.base_img_size[0],
                img_size.shape[1] / self.base_img_size[1],
            )

            noise = torch.randn_like(x_start) * noise_mul

        assert noise.shape == x_start.shape
        return self.mollifier(
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, noise.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior: q(x_{t-1} | x_t, x_0)
        Note that this method has nothing to do with the shapes of x_start and x_t
        """
        assert x_start.shape == x_t.shape

        posterior_x_start_component = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
        )
        posterior_x_t_component = (
            extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_mean = posterior_x_start_component + posterior_x_t_component

        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return (
            posterior_mean,
            posterior_variance,
            posterior_log_variance_clipped,
            posterior_x_start_component,
            posterior_x_t_component,
        )

    def p_mean_variance(
        self, model, x, t, clip_denoised=False, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of the initial x_0.

        :param clip_denoised: if True, clip the ''denoised signal'' into designated value.
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        model_variance, model_log_variance = {
            ModelVarType.FIXED_LARGE: (
                np.append(self.posterior_variance[1], self.betas[1:]),
                np.log(np.append(self.posterior_variance[1], self.betas[1:])),
            ),
            ModelVarType.FIXED_SMALL: (
                self.posterior_variance,
                self.posterior_log_variance_clipped,
            ),
        }[self.model_var_type]
        model_variance = extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(self.clip_min, self.clip_max)
            return x

        if self.model_mean_type == ModelMeanType.MOLLIFIED_EPSILON:
            # For ModelMeanType.MOLLIFIED_EPSILON this is actually Tx_0 instead of x_0
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
            )
            model_mean, _, _, posterior_x_start_component, posterior_x_t_component = (
                self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert model_mean.shape == pred_xstart.shape == x.shape
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            "posterior_x_start_component": posterior_x_start_component,
            "posterior_x_t_component": posterior_x_t_component,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        In particular, cond_fn computes grad(log(p(y|x))), and we want to condition on y.
        """
        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        alpha_bar = extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
            x, self._scale_timesteps(t), **model_kwargs
        )

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _, _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        # todo Suspect this place source code has an error

        return out

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=False,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        noise_mul=1.0,  # can be changed
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = torch.randn_like(x) * noise_mul
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0

        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )

        if self.model_mean_type == ModelMeanType.MOLLIFIED_EPSILON:
            sample = out["mean"] + nonzero_mask * torch.exp(
                0.5 * out["log_variance"]
            ) * self.mollifier(noise)
        else:
            raise ValueError("ModelMeanType not MOLLIFIED_EPSILON")

        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=False,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        noise_mul=1.0,  # can be changed
    ):
        """
        Generate samples from the model and yield intermediate samples from each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of p_sample().
        """
        if device is None:
            device = next(model.parameters()).device

        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=device)
            img = self.mollifier(img * noise_mul)

        indices = list(range(self.num_timesteps))[::-1]

        if model_kwargs is None:
            model_kwargs = {}

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    noise_mul=noise_mul,
                )
                yield out
                img = out["sample"]

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=False,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        return_all=False,
        noise_mul=1.0,  # can be changed
    ):
        """
        Generate samples from the model.

        :param shape: the shape of the samples, (N, C, H, W).
        :param clip_denoised: if True, clip x_start predictions to designated value.
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        all_samples = []
        all_pred_xstarts = []
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            noise_mul=noise_mul,
        ):
            final = sample
            if return_all:
                all_samples.append(sample["sample"][0].float().cpu())
                all_pred_xstarts.append(sample["pred_xstart"][0].float().cpu())

        if return_all:
            return (
                final["sample"],
                final["pred_xstart"],
                torch.stack(all_samples),
                torch.stack(all_pred_xstarts),
            )
        else:
            return final["sample"], final["pred_xstart"]

    def SDEdit(
        self,
        model,
        x_0,
        t_0,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        noise_mul=1.0,
    ):
        """
        Perform SDEdit: add noise to a given sample and then denoise to refine it.

        :param model: The model to use for reverse denoising.
        :param x_0: The initial sample, a tensor of shape [B, C, H, W].
        :param t_0: The noise level to add, a float between 0 and 1 indicating the degree of forward noise.
        :param clip_denoised: Whether to clip the denoised samples during the process.
        :param denoised_fn: Optional function to modify x_start predictions before use.
        :param cond_fn: Optional gradient-based conditioning function.
        :param model_kwargs: Optional dict of extra keyword arguments for the model.
        :param noise_mul: Multiplier for noise during sampling.
        :return: The final denoised sample, refined from the original input.
        """
        # Ensure t_0 is valid
        assert 0 <= t_0 <= 1, "t_0 must be a float between 0 and 1."

        if model_kwargs is None:
            model_kwargs = {}

        # Convert t_0 from 0-1 to a timestep in the diffusion process
        t_0_step = int(t_0 * (self.num_timesteps - 1))
        t_0_tensor = torch.tensor([t_0_step] * x_0.size(0), device=x_0.device)

        # Add noise to x_0 using q_sample
        noise = torch.randn_like(x_0) * noise_mul
        noise = noise.to(x_0.device)
        x_t = self.q_sample(x_0, t_0_tensor, noise=noise)

        # Prepare for reverse denoising
        img = x_t  # Start from the noisy image
        indices = list(range(t_0_step, -1, -1))  # Reverse from t_0_step to 0

        # Iteratively denoise
        for i in indices:
            t = torch.tensor([i] * x_t.size(0), device=x_0.device)
            with torch.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    noise_mul=noise_mul,
                )

                img = out["sample"]

        return img

    def Inpaint(
        self,
        model,
        x_0,
        t_0,
        true_x_0,
        mask,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        noise_mul=1.0,
        rewinding_step=25,
        rewinding_round=10,
        repainting_time_interval=5,
    ):
        """
        Perform inpainting using SDEdit with guidance from a given mask.

        :param model: The model to use for reverse denoising.
        :param x_0: The initial sample, a tensor of shape [B, C, H, W].
        :param t_0: The noise level to add, a float between 0 and 1 indicating the degree of forward noise.
        :param true_x_0: The target ground truth. Must be a tensor.
        :param mask: A tensor of shape [B, C, H, W], with 0 indicating regions to be inpainted, and 1 indicating observation-guided regions. Must be a tensor.
        :param clip_denoised: Whether to clip denoised values during reverse diffusion.
        :param denoised_fn: Optional function to modify x_start predictions before use.
        :param cond_fn: Optional gradient-based conditioning function.
        :param model_kwargs: Optional dictionary of extra arguments for the model.
        :param noise_mul: Multiplier for the noise added during sampling.
        :param rewinding_step: Number of timesteps to rewind during repainting.
        :param rewinding_round: Number of rewinding rounds per repainting point.
        :param repainting_time_interval: Interval at which repainting is applied.
        :return: The final denoised sample, refined through observation-guided inpainting.
        """
        assert 0 <= t_0 <= 1, "t_0 must be a float between 0 and 1."

        if model_kwargs is None:
            model_kwargs = {}

        true_x_0 = true_x_0.to(x_0.device)
        mask = mask.to(x_0.device)

        # Convert t_0 from a float in [0,1] to a discrete timestep
        t_0_step = int(t_0 * (self.num_timesteps - 1))
        t_0_tensor = torch.tensor([t_0_step] * x_0.size(0), device=x_0.device)

        # Add forward noise to x_0
        noise = torch.randn_like(x_0) * noise_mul
        noise = noise.to(x_0.device)
        x_t = self.q_sample(x_0, t_0_tensor, noise=noise)

        # Initialize image to noisy input
        img = x_t
        indices = list(range(t_0_step, -1, -1))  # Timesteps from t_0_step down to 0

        # Select timesteps where repainting will be applied
        omega = [
            i
            for i in indices
            if i % repainting_time_interval == 0
            and i != 0
            and i > (t_0_step - 40)  # Important to change to test effect
        ]

        # Precompute random noise used in guided sampling for each timestep
        result_dict = {
            i: (torch.randn_like(true_x_0) * noise_mul).to(x_0.device)
            for i in range(1000)
        }

        # Reverse diffusion loop
        for i in indices:
            t = torch.tensor([i] * x_t.size(0), device=x_0.device)

            with torch.no_grad():
                # Standard reverse denoising step
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    noise_mul=noise_mul,
                )

                img = out["sample"]

                if i - 1 in omega:
                    for _ in range(rewinding_round):
                        # Rewind: inject noise into the denoised image
                        img = np.sqrt(
                            self.compute_alpha_prod(i - 1, i + rewinding_step - 1)
                        ) * img + (
                            np.sqrt(
                                1
                                - self.compute_alpha_prod(i - 1, i + rewinding_step - 1)
                            )
                        ) * self.mollifier(
                            (torch.randn_like(true_x_0) * noise_mul).to(x_0.device)
                        )

                        # Generate noisy version of true_x_0 at the rewinding step
                        mask_true = self.q_sample(
                            true_x_0,
                            t - 1 + rewinding_step,
                            noise=result_dict[i - 1 + rewinding_step],
                        )

                        # Apply the observation mask to guide the current sample
                        img = img * (1 - mask) + mask_true * mask

                        # Re-apply reverse denoising from the rewinding point
                        for j in range(i + rewinding_step - 1, i - 1, -1):
                            out = self.p_sample(
                                model,
                                img,
                                torch.tensor([j] * x_t.size(0), device=x_0.device),
                                clip_denoised=clip_denoised,
                                denoised_fn=denoised_fn,
                                cond_fn=cond_fn,
                                model_kwargs=model_kwargs,
                                noise_mul=noise_mul,
                            )

                            img = out["sample"]

        return img

    def training_losses(
        self,
        model,
        x_start,
        t,
        sample_lst=None,
        model_kwargs=None,
        noise=None,
    ):
        """
        Compute training losses for a single timestep.

        :param x_start: shape B,C,H,W
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
        """
        if model_kwargs is None:
            model_kwargs = {}

        device = x_start.device

        img_size = (x_start.size(-2), x_start.size(-1))  # Extract H and W respectively

        if img_size != self.base_img_size:
            x_start = x_start.cpu().numpy()
            if isinstance(x_start, np.ndarray):
                zoom_factors = [1, 1] + [
                    n / o for n, o in zip(self.base_img_size, img_size)
                ]
                x_start = zoom(x_start, zoom=zoom_factors, order=0)
            else:
                raise TypeError("x_start must be a numpy.ndarray")

            x_start = torch.from_numpy(x_start).to(device)

        model_kwargs["base_img_size"] = self.base_img_size

        # Compute noise multiplier; noise_mul should be a scalar
        noise_mul = min(
            x_start.size(-2) / self.base_img_size[0],
            x_start.size(-1) / self.base_img_size[1],
        )

        if noise is None:
            noise = torch.randn_like(x_start) * noise_mul

        x_t = self.q_sample(x_start, t, noise=noise)
        # Note: the sampled x_t has already undergone mollification

        mollified_noise = None
        if self.model_mean_type == ModelMeanType.MOLLIFIED_EPSILON:
            mollified_noise = self.mollifier(noise)

        terms = {}  # The dictionary to be returned by this function

        if sample_lst is not None:
            model_kwargs["sample_lst"] = sample_lst
            x_t = rearrange(x_t, "b c h w -> b (h w) c")
            # print("x_t shape", x_t.shape)
            x_t = torch.gather(
                x_t, 1, sample_lst.unsqueeze(2).repeat(1, 1, x_t.size(2))
            ).contiguous()

            # Note: x_start and noise do not need subsampling here
            # ---------------------------------------------------------------------
            # x_start = rearrange(x_start, "b c h w -> b (h w) c")
            # x_start = torch.gather(
            #     x_start, 1, sample_lst.unsqueeze(2).repeat(1, 1, x_start.size(2))
            # ).contiguous()
            # noise = rearrange(noise, "b c h w -> b (h w) c")
            # noise = torch.gather(
            #     noise, 1, sample_lst.unsqueeze(2).repeat(1, 1, noise.size(2))
            # ).contiguous()
            # ---------------------------------------------------------------------

            if mollified_noise is not None:
                mollified_noise = rearrange(mollified_noise, "b c h w -> b (h w) c")
                mollified_noise = torch.gather(
                    mollified_noise,
                    1,
                    sample_lst.unsqueeze(2).repeat(1, 1, mollified_noise.size(2)),
                ).contiguous()

        if self.loss_type == LossType.MSE or self.loss_type == LossType.L1:
            model_output = model(
                x_t, self._scale_timesteps(t), **model_kwargs
            )  # Note: model_kwargs includes z

            target = {
                ModelMeanType.MOLLIFIED_EPSILON: mollified_noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape

            if self.loss_type == LossType.MSE:
                terms["loss"] = mean_flat((target - model_output) ** 2)
            else:
                terms["loss"] = mean_flat(torch.abs(target - model_output))
        else:
            raise NotImplementedError(self.loss_type)

        return terms
