
# sun by: python experiment/generate_samples.py

# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from absl import app # type: ignore
from absl import flags # type: ignore
from ml_collections.config_flags import config_flags  # type: ignore
import os
from tqdm import tqdm
import numpy as np

from models import SparseUNet
import diffusion_class.diffusion_util as gd
from diffusion_class import (
    GaussianDiffusion,
    get_named_beta_schedule
)
from dct_util import DCTGaussianBlur
from utils import BasePrecipitationDataset

# Commandline arguments
FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config",
    "configs/training_config.py", # "/configs/training_config.py" That's the wrong way to write it
    "configuration.",
    lock_config=True,
)
# flags.mark_flags_as_required(["config"])

# Torch options
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


def main(argv):

    H = FLAGS.config
    sample_img_size = H.generation.sample_img_size
    rank = H.run.gpu
    device = torch.device(f"cuda:{rank}")
    # device = torch.device("cuda:0")
    
    base_img_height, base_img_width = H.data.img_size
    model = SparseUNet(
        channels=H.data.channels,
        nf=H.model.nf,
        time_emb_dim=H.model.time_emb_dim,
        img_size=(base_img_height, base_img_width),
        num_conv_blocks=H.model.num_conv_blocks,
        knn_neighbours=H.model.knn_neighbours,
        uno_res=H.model.uno_res,
        uno_mults=H.model.uno_mults,
        z_dim=H.model.z_dim,
        conv_type=H.model.uno_conv_type,
        depthwise_sparse=H.model.depthwise_sparse,
        kernel_size=H.model.kernel_size,
        backend=H.model.backend,
        blocks_per_level=H.model.uno_blocks_per_level,
        attn_res=H.model.uno_attn_resolutions,
        dropout_res=H.model.uno_dropout_from_resolution,
        dropout=H.model.uno_dropout,
        uno_base_nf=H.model.uno_base_channels,
    )
    
    checkpoint_path = f"./checkpoints/{H.run.experiment}/checkpoint_step_{H.train.checkpoint_num}_rank_0.pkl"
    
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    print(f"Loading model from step {state_dict['global_step']}")

    model.load_state_dict(state_dict["model_ema_state_dict"])
    
    model = model.to(device)

    betas = get_named_beta_schedule(
        H.diffusion.noise_schedule,
        H.diffusion.steps,
    )
    
    if H.diffusion.model_mean_type == "epsilon":
        model_mean_type = gd.ModelMeanType.EPSILON
    elif H.diffusion.model_mean_type == "xstart":
        model_mean_type = gd.ModelMeanType.START_X
    elif H.diffusion.model_mean_type == "mollified_epsilon":
        assert (
            H.diffusion.gaussian_filter_std > 0
        ), "Error: Predicting mollified_epsilon but gaussian_filter_std == 0."
        model_mean_type = gd.ModelMeanType.MOLLIFIED_EPSILON
    else:
        raise Exception(
            "Unknown model mean type. Expected value in [epsilon, mollified_epsilon, xstart]"
        )

    model_var_type = (
        gd.ModelVarType.FIXED_LARGE
        if not H.model.sigma_small
        else gd.ModelVarType.FIXED_SMALL
    )

    loss_type = gd.LossType.MSE if H.diffusion.loss_type == "MSE" else gd.LossType.L1

    model_diffusion = GaussianDiffusion(
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        H.diffusion.gaussian_filter_std,
        img_size=(base_img_height, base_img_width),
        rescale_timesteps=False,
        mollifier_type=H.diffusion.mollifier_type,
        clip_min=H.data.clip_min,
        clip_max=H.data.clip_max,
    ).to(device)
    
    dataset = BasePrecipitationDataset(H)
         
    make_samples(
        H,
        model,
        model_diffusion,
        dataset=dataset,
        device=device,
        sample_img_size=sample_img_size,
    )



def make_samples(
    H,
    model,
    diffusion,
    *,
    dataset=None,
    device=None,
    sample_img_size=None, 
):

    if sample_img_size is not None:
        # The scaling ratios in the height and width directions are calculated separately
        std_ratio_H = sample_img_size[0] / H.data.img_size[0]
        std_ratio_W = sample_img_size[1] / H.data.img_size[1]

        # Calculate the new standard deviation ratio and pass it to DCTGaussianBlur
        new_std_ratio = min(std_ratio_H, std_ratio_W)

        if new_std_ratio == 1:
            print("new_std_ratio == 1, just use original DCTGaussianBlur")
        else:
            print("new_std_ratio != 1, use new DCTGaussianBlur")
            diffusion.mollifier = DCTGaussianBlur(
                sample_img_size, std=new_std_ratio * H.diffusion.gaussian_filter_std
            ).to(device)

    unique_name = H.run.unique_name
    
    if unique_name == None:
        save_dir = f"checkpoints/{H.run.experiment}_samples_{sample_img_size}_checkpoints_{H.train.checkpoint_num}_device_{H.run.gpu}"
    else:
        save_dir = f"checkpoints/{H.run.experiment}_samples_{sample_img_size}_checkpoints_{H.train.checkpoint_num}__device_{H.run.gpu}_{unique_name}"

    os.makedirs(
        save_dir,
        exist_ok=True,
    )

    # The noise multiplier is calculated separately,noise mul should be a scalar
    noise_mul = min(
        sample_img_size[0] / H.data.img_size[0], sample_img_size[1] / H.data.img_size[1]
    )

    idx = 0
    for _ in tqdm(range(H.data.fid_samples // H.generation.sample_size + 1)):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=H.train.amp):
                samples, _ = diffusion.p_sample_loop(
                    model,
                    (
                        H.generation.sample_size,
                        H.data.channels,
                        sample_img_size[0],
                        sample_img_size[1],
                    ),
                    clip_denoised=True,
                    progress=True,
                    model_kwargs=dict(
                        z=(H.generation.idx)
                        .unsqueeze(0)
                        .repeat(H.generation.sample_size, 1)
                        .to(device)
                    ),
                    return_all=False,
                    noise_mul=noise_mul,
                )
        

                def save_to_npy(array, filename):
                    np.save(os.path.join(save_dir, filename), array)

                if H.diffusion.model_mean_type == "mollified_epsilon":
                    deblurred_samples = diffusion.mollifier.undo_wiener(samples)
                    deblurred_samples = dataset.apply_denormalization(deblurred_samples)

                samples = dataset.apply_denormalization(samples)
                
                save_to_npy(
                    samples,
                    filename=f"samples_{samples.shape[0]}x{samples.shape[2]}x{samples.shape[3]}_idx_{idx}.npy",
                )

                if H.diffusion.model_mean_type == "mollified_epsilon":
                    save_to_npy(
                        deblurred_samples,
                        filename=f"deblurred_samples_{deblurred_samples.shape[0]}x{deblurred_samples.shape[2]}x{deblurred_samples.shape[3]}_idx_{idx}.npy",
                    )

                idx = idx + 1

                if idx == (H.data.fid_samples // H.generation.sample_size + 1):
                    break

                
                
if __name__ == "__main__":
    app.run(main)
