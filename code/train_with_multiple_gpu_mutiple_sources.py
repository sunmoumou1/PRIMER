import torch  # type: ignore
import torch.distributed as dist  # type: ignore
import torch.multiprocessing as mp  # type: ignore
from torch.nn.parallel import DistributedDataParallel as DDP  # type: ignore
import numpy as np
from torch import isnan  # type: ignore

import wandb  # type: ignore
from absl import app  # type: ignore
from absl import flags  # type: ignore

from ml_collections.config_flags import config_flags  # type: ignore
import time
import os
import random
from itertools import cycle
from models import SparseUNet
from utils import (
    get_data_loader,
    flatten_collection,
    optim_warmup,
    update_ema,
    create_named_schedule_sampler,
    LossAwareSampler,
    optim_decay,
    upscale_sample_indices,
)
from diffusion_class.diffusion_util import get_named_beta_schedule
import diffusion_class.diffusion as gd
from diffusion_class.diffusion import GaussianDiffusion

# Commandline arguments
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config",
    "./configs/training_config.py",
    "Training configuration.",
    lock_config=True,
)

# •	"config" is the name of the parameter, and you can specify the configuration file using --config on the command line.
# •	"Training configuration." is the description of this parameter, which will be displayed when the user requests help information.
# •	lock_config=True means that the configuration cannot be modified after it is loaded, which helps prevent accidental changes during runtime.

# ----------------------------------------------------

# Torch options
torch.backends.cuda.matmul.allow_tf32 = True
# TF32 is a floating-point format introduced on NVIDIA GPUs with the Ampere architecture. It offers a balance between computational speed and numerical precision. Enabling TF32 can significantly accelerate the training of deep learning models while maintaining sufficient numerical accuracy.

torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


def print_module_params(model, module_names):
    """
    Print the total number of trainable parameters in the model,
    as well as the number of trainable parameters in specified modules.

    Args:
    - model (torch.nn.Module): The deep learning model to analyze.
    - module_names (list of str): A list of module names within the model
      for which to count the number of parameters.

    Example usage:
    module_names = ['down_blocks', 'up_blocks', 'uno']
    print_module_params(model, module_names)

    Note:
    - This function only counts parameters where `requires_grad=True`,
      representing those actually involved in training.
    """

    # Compute the total number of trainable parameters in the model
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters in the entire model: {num_trainable_params}")
    print("---" * 20)

    # Iterate over the specified modules and compute parameter counts
    for module_name in module_names:
        module = getattr(model, module_name, None)
        if module is not None:
            num_module_params = sum(
                p.numel() for p in module.parameters() if p.requires_grad
            )
            print(
                f"Number of trainable parameters in {module_name}: {num_module_params}"
            )
        else:
            print(f"Module {module_name} not found in the model.")


def setup_ddp(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"  # Select a port that is free

    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=world_size, rank=rank
    )


def cleanup_ddp():
    """
    Clean up distributed process groups
    """
    dist.destroy_process_group()


def generate_samples(
    num_epoch=None,
    rank=None,
    model=None,
    diffusion=None,
    dataset=None,
    H=None,
    output_folder=None,
):
    if rank == 0:
        z = (
            torch.tensor(
                [0, 0, 1], dtype=torch.float32
            )  # you can change the style here!!!
            .unsqueeze(0)
            .repeat(H.generation.sample_size, 1)
            .to(torch.device(f"cuda:{rank}"))
        )
    elif rank == 1:
        z = (
            torch.tensor(
                [0, 0, 1], dtype=torch.float32
            )  # choose whatever style you would like to generate
            .unsqueeze(0)
            .repeat(H.generation.sample_size, 1)
            .to(torch.device(f"cuda:{rank}"))
        )

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=H.train.amp):
            # Define different image sizes for generation
            sizes = [
                (
                    H.generation.sample_img_size[0],
                    H.generation.sample_img_size[1],
                )
            ] * H.generation.sample_num  # repeated samples at original resolution

            all_samples = []
            all_deblurred_samples = []

            # Generate samples for each target image size
            for size in sizes:
                h, w = size

                noise_mul = min(
                    h / H.data.expected_img_size[-2], w / H.data.expected_img_size[-1]
                )

                samples, _ = diffusion.p_sample_loop(
                    model,
                    (
                        H.generation.sample_size,
                        H.data.channels,
                        h,
                        w,
                    ),
                    clip_denoised=True,
                    progress=True if rank == 0 else False,
                    model_kwargs=dict(z=z),
                    return_all=False,  # Return only the final samples, not all intermediate steps
                    noise_mul=noise_mul,
                )

                if H.diffusion.model_mean_type == "mollified_epsilon":
                    deblurred_samples = diffusion.mollifier.undo_wiener(samples)
                    deblurred_samples = dataset.apply_denormalization(deblurred_samples)
                    all_deblurred_samples.append(deblurred_samples)

                samples = dataset.apply_denormalization(samples)
                all_samples.append(samples)

            def save_to_npy(array, filename):
                """
                Save a tensor or ndarray as a .npy file.

                Args:
                    array: A NumPy ndarray or PyTorch tensor to save.
                    filename: Name of the output file.
                """
                if isinstance(array, torch.Tensor):
                    array = array.cpu().numpy()
                elif not isinstance(array, np.ndarray):
                    raise ValueError("Input must be a NumPy ndarray or PyTorch tensor.")

                os.makedirs(output_folder, exist_ok=True)
                np.save(os.path.join(output_folder, filename), array)

            # Concatenate all samples along the first dimension and save
            all_samples = np.concatenate(all_samples, axis=0)
            save_to_npy(
                all_samples,
                filename=f"samples_epoch_{num_epoch}_{h}x{w}_rank_{rank}.npy",
            )

            if H.diffusion.model_mean_type == "mollified_epsilon":
                all_deblurred_samples = np.concatenate(all_deblurred_samples, axis=0)
                save_to_npy(
                    all_deblurred_samples,
                    filename=f"deblurred_samples_epoch_{num_epoch}_{h}x{w}_rank_{rank}.npy",
                )


def train_on_batch(
    batch,
    *,
    H,
    global_step,
    initial_global_step,
    optim,
    schedule_sampler,
    model,
    ema_model,
    diffusion,
    scaler,
    rank,
    mean_loss,
    mean_loss_era5,
    mean_loss_imerg,
    mean_loss_gauge,
    mean_step_time,
    skip,
    mean_total_norm,
    q_sample_ratio,
    identifier,
):
    """
    All arguments except `batch` must be passed as keyword arguments.
    """
    start_time = time.time()

    def unpack_batch(batch):
        """
        Unpack the batch based on the number of elements.

        :param batch: Tuple, can be (x, idx, mask) or (x, idx)
        :return: x, idx, mask (default is None if not provided)
        """
        if len(batch) == 3:
            x, idx, mask = batch
        elif len(batch) == 2:
            x, idx, mask = batch[0], batch[1], None
        else:
            raise ValueError("Batch must be a tuple with 2 or 3 elements.")
        return x, idx, mask

    x, idx, mask = unpack_batch(batch)

    img_height, img_width = x.shape[-2], x.shape[-1]
    batch_size = x.shape[0]

    device = torch.device(f"cuda:{rank}")

    if (global_step - initial_global_step) < H.optimizer.warmup_steps:
        optim_warmup(
            (global_step - initial_global_step),
            optim,
            H.optimizer.learning_rate,
            H.optimizer.warmup_steps,
        )

    x = x.to(device, non_blocking=True)
    # non_blocking=True allows asynchronous data transfer to improve training efficiency.

    t, weights = schedule_sampler.sample(batch_size, device)
    # schedule_sampler samples time steps t for each sample and provides corresponding weights.

    if H.mc_integral.type == "uniform":
        if mask is None:
            sample_lst = torch.stack(
                [
                    torch.from_numpy(
                        np.random.choice(
                            img_height * img_width,
                            int(img_height * img_width * q_sample_ratio),
                            replace=False,
                        )
                    )
                    for _ in range(batch_size)
                ]
            ).to(device)
        else:
            sample_lst = torch.stack(
                [
                    torch.from_numpy(
                        np.random.choice(
                            torch.where(mask[b].flatten() == 0)[0].numpy(),
                            int(img_height * img_width * q_sample_ratio),
                            replace=False,
                        )
                    )
                    for b in range(batch_size)
                ]
            ).to(device)
    else:
        raise Exception("Unknown Monte Carlo Integral type")

    with torch.cuda.amp.autocast(enabled=H.train.amp):

        if (1, img_height, img_width) != H.data.expected_img_size:
            sample_lst = upscale_sample_indices(
                sample_lst,
                (img_height, img_width),
                (H.data.expected_img_size[-2], H.data.expected_img_size[-1]),
            ).to(device)

        losses = diffusion.training_losses(
            model, x, t, sample_lst=sample_lst, model_kwargs={"z": idx}
        )

        # Check if losses["loss"] contains NaNs
        loss_values = losses["loss"]
        if torch.isnan(loss_values).any():
            print('losses["loss"] contains NaN values.')
            print("Current loss values:", loss_values)

            valid_mask = ~torch.isnan(loss_values)
            loss_values = loss_values[valid_mask]
            valid_weights = weights[valid_mask]
        else:
            valid_weights = weights

        loss = (loss_values * valid_weights).mean()

        if isnan(loss):
            print("Final computed loss is still NaN. Skipping this batch.")
            optim.zero_grad()
            dist.barrier()
            return mean_loss, mean_step_time, skip, mean_total_norm

    optim.zero_grad()
    if H.train.amp:
        scaler.scale(loss).backward()
        scaler.unscale_(optim)

        model_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if (
            H.optimizer.gradient_skip
            and model_total_norm >= H.optimizer.gradient_skip_threshold
        ):
            print(f"model_total_norm: {model_total_norm}, skipping parameter update.")
            scaler.update()
            skip += 1
        else:
            scaler.step(optim)
            scaler.update()
    else:
        loss.backward()
        model_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if (
            H.optimizer.gradient_skip
            and model_total_norm >= H.optimizer.gradient_skip_threshold
        ):
            skip += 1
        else:
            optim.step()

    if isinstance(schedule_sampler, LossAwareSampler):
        schedule_sampler.update_with_local_losses(t, losses["loss"].detach())

    if global_step % H.train.ema_update_every == 0:
        update_ema(model, ema_model, H.train.ema_decay)

    mean_loss += loss.item()
    if identifier == 0:
        mean_loss_era5 += loss.item()
    elif identifier == 1:
        mean_loss_imerg += loss.item()
    else:
        mean_loss_gauge += loss.item()

    mean_step_time += time.time() - start_time
    mean_total_norm += model_total_norm.item()

    if rank == 0:
        wandb_dict = dict()

    if global_step % H.train.plot_graph_steps == 0:
        norm = H.train.plot_graph_steps

        if rank == 0:
            print(
                f"Step: {global_step}, Loss {mean_loss / norm:.5f}, Loss_era5 {mean_loss_era5 / H.train.sources_proportion[0] / norm:.5f}, Loss_imerg {mean_loss_imerg / H.train.sources_proportion[1]  / norm:.5f}, Loss_gauge {mean_loss_gauge / H.train.sources_proportion[2]  / norm:.5f}, Step Time: {mean_step_time / norm:.5f}, Skip: {skip / norm:.5f}, Gradient Norm: {mean_total_norm / norm:.5f}"
            )
            wandb_dict |= {
                "Step Time": mean_step_time / norm,
                "Loss": mean_loss / norm,
                "Loss_era5": mean_loss_era5 / norm / H.train.sources_proportion[0],
                "Loss_imerg": mean_loss_imerg / norm / H.train.sources_proportion[1],
                "Loss_gauge": mean_loss_gauge / norm / H.train.sources_proportion[2],
                "Skip": skip / norm,
                "Gradient Norm": mean_total_norm / norm,
            }

        mean_loss = 0
        mean_loss_era5 = 0
        mean_loss_imerg = 0
        mean_loss_gauge = 0

        mean_step_time = 0
        skip = 0
        mean_total_norm = 0

    if rank == 0:
        if wandb_dict:
            wandb.log(wandb_dict, step=global_step)

    dist.barrier()

    return (
        mean_loss,
        mean_loss_era5,
        mean_loss_imerg,
        mean_loss_gauge,
        mean_step_time,
        skip,
        mean_total_norm,
    )


def train(
    H,
    model,
    ema_model,
    train_loaders,  # train_loaders is a list of DataLoaders
    optim,
    diffusion,
    schedule_sampler,
    rank,  # This parameter is especially important
    checkpoint_path="",
    global_step=0,
):
    scaler = torch.cuda.amp.GradScaler()

    mean_loss = 0
    mean_loss_era5 = 0
    mean_loss_imerg = 0
    mean_loss_gauge = 0
    mean_step_time = 0
    mean_total_norm = 0
    skip = 0

    batch_size = H.train.batch_size

    initial_global_step = global_step

    # Create subfolder to save sampled images
    output_folder = f"sampling_image/{H.run.experiment}"
    os.makedirs(output_folder, exist_ok=True)

    if rank == 0:
        print(
            "---------------------------------- Start Training ----------------------------------"
        )

    # Training loop
    num_epoch = 0
    if H.train.retrain_epoch is not None:
        num_epoch = H.train.retrain_epoch
        print(f"Resuming training from epoch {num_epoch}")

    while True:
        prob1 = H.train.sources_proportion[0]  # change proportion whatever you like
        prob2 = H.train.sources_proportion[0] + H.train.sources_proportion[1]

        for _ in range(
            len(train_loaders[0])
        ):  # The length of the first DataLoader is used as the baseline
            # Constructing the current batch
            batch_data = []
            batch_labels = []
            batch_masks = []

            random_number = random.random()

            if random_number < prob1:
                identifier = 0
                q_sample_ratio = H.mc_integral.q_sample_ratio[0]
                for _ in range(batch_size):
                    data, label, mask = train_loaders[0].dataset[0]

                    # Populate data to a uniform format
                    batch_data.append(data)
                    batch_labels.append(label)
                    batch_masks.append(mask)
            elif random_number < prob2:
                identifier = 1
                q_sample_ratio = H.mc_integral.q_sample_ratio[1]
                for _ in range(batch_size):
                    data, label, mask = train_loaders[1].dataset[0]

                    # Populate data to a uniform format
                    batch_data.append(data)
                    batch_labels.append(label)
                    batch_masks.append(mask)
            else:
                identifier = 2
                q_sample_ratio = H.mc_integral.q_sample_ratio[2]
                for _ in range(batch_size):
                    data, label, mask = train_loaders[2].dataset[0]

                    # Populate data to a uniform format
                    batch_data.append(data)
                    batch_labels.append(label)
                    batch_masks.append(mask)

            # Combine batches into a single batch
            combined_batch = (
                torch.stack(batch_data),
                torch.stack(batch_labels),
                torch.stack(batch_masks),
            )

            global_step += 1
            (
                mean_loss,
                mean_loss_era5,
                mean_loss_imerg,
                mean_loss_gauge,
                mean_step_time,
                skip,
                mean_total_norm,
            ) = train_on_batch(
                combined_batch,
                H=H,
                global_step=global_step,
                initial_global_step=initial_global_step,
                optim=optim,
                schedule_sampler=schedule_sampler,
                model=model,
                ema_model=ema_model,
                diffusion=diffusion,
                scaler=scaler,
                rank=rank,
                mean_loss=mean_loss,
                mean_loss_era5=mean_loss_era5,
                mean_loss_imerg=mean_loss_imerg,
                mean_loss_gauge=mean_loss_gauge,
                mean_step_time=mean_step_time,
                skip=skip,
                mean_total_norm=mean_total_norm,
                q_sample_ratio=q_sample_ratio,
                identifier=identifier,
            )

        num_epoch += 1

        # Save checkpoint at the end of each epoch
        checkpoint_path_new = os.path.join(
            checkpoint_path,
            f"checkpoint_step_{global_step}_rank_{rank}.pkl",
        )
        # Only save model on rank 0
        if rank == 0:
            torch.save(
                {
                    "global_step": global_step,
                    "model_state_dict": (
                        model.module.state_dict()
                        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
                        else model.state_dict()
                    ),
                    "model_ema_state_dict": (
                        ema_model.module.state_dict()
                        if isinstance(
                            ema_model, torch.nn.parallel.DistributedDataParallel
                        )
                        else ema_model.state_dict()
                    ),
                    "optimizer_state_dict": optim.state_dict(),
                },
                checkpoint_path_new,
            )
            print(
                f"Checkpoint saved at step {global_step} (epoch {num_epoch}) for rank {rank} at {checkpoint_path_new}"
            )

        dist.barrier()

        # Generate and save samples at the end of each epoch
        # you can change generate_samples function to decide which style you want to check during training
        generate_samples(
            num_epoch,
            rank,
            ema_model.module,
            diffusion,
            train_loaders[0].dataset.dataset,
            H,
            output_folder,
        )
        if rank == 0:
            print(f"Rank {rank}: successfully generated samples")
            print("*" * 100)

        # Check if learning rate decay is needed
        if num_epoch % H.optimizer.decay_interval == 0:
            # Apply learning rate decay
            optim_decay(optim, minimum_lr=H.optimizer.minimum_lr)

            if rank == 0:
                current_lr = optim.param_groups[0]["lr"]
                print(f"Current learning rate after decay: {current_lr}")
                print("*" * 100)

        if rank == 0:
            print(f"Finished training epoch {num_epoch}")
            print("*" * 100)


def main_worker(rank, world_size, H):
    torch.cuda.set_device(rank)

    setup_ddp(rank, world_size)

    device = torch.device(f"cuda:{rank}")

    print("This process local_rank is:", rank)
    print("This process device is:", device)
    print("---" * 20)

    if rank == 0:
        # wandb can be disabled by passing in --config.run.wandb_mode=disabled
        wandb.init(
            project=H.run.name,
            config=flatten_collection(
                H
            ),  # Flatten the config H before passing to W&B for logging.
            save_code=True,
            dir=H.run.wandb_dir,
            mode=H.run.wandb_mode,  # online
        )

    base_img_height, base_img_width = (
        H.data.expected_img_size[-2],
        H.data.expected_img_size[-1],
    )

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

    ema_model = SparseUNet(
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

    module_names = ["down_blocks", "up_blocks", "uno"]
    if rank == 0:
        print_module_params(model, module_names)

    if H.run.experiment != "":
        checkpoint_path = f"checkpoints/{H.run.experiment}/"
    else:
        checkpoint_path = "checkpoints/"

    if rank == 0:
        os.makedirs(checkpoint_path, exist_ok=True)

    train_kwargs = {}
    train_kwargs["checkpoint_path"] = checkpoint_path

    # Move models to the corresponding GPU device
    model = model.to(device)
    ema_model = ema_model.to(device)

    train_loaders, _ = get_data_loader(
        H,
        rank=rank,
        world_size=world_size,
    )

    def compute_learning_rate(initial_lr, decay_interval, retrain_epoch):
        """
        Compute the learning rate during fine-tuning based on the decay strategy.

        Parameters:
        - initial_lr (float): Initial learning rate.
        - decay_interval (int): Number of epochs after which the learning rate is halved.
        - retrain_epoch (int): Epoch from which fine-tuning starts.

        Returns:
        - float: Adjusted learning rate for fine-tuning.
        """
        if retrain_epoch is None:
            return initial_lr

        if retrain_epoch < decay_interval:
            return initial_lr

        num_decays = retrain_epoch // decay_interval
        current_lr = initial_lr * (0.5**num_decays)
        return current_lr

    optim = torch.optim.AdamW(
        list(model.parameters()),
        lr=H.optimizer.learning_rate,
        weight_decay=H.optimizer.weight_decay,
    )

    # Dynamically set checkpoint path for each rank
    checkpoint_path_new = f"./checkpoints/{H.run.experiment}/checkpoint_step_{H.train.checkpoint_num}_rank_0.pkl"
    if H.train.load_checkpoint and os.path.exists(checkpoint_path_new):
        # Load checkpoint file for rank 0
        state_dict = torch.load(checkpoint_path_new, map_location=device)

        # Set global_step
        train_kwargs["global_step"] = state_dict["global_step"]
        if rank == 0:
            print(f"Rank {rank}: Loading model from step {state_dict['global_step']}")

        if H.train.whether_global_step_zero is True:
            train_kwargs["global_step"] = 0
            print(f"Rank {rank}: Overwriting step to 0 after loading")

        # Load model weights
        model.load_state_dict(state_dict["model_state_dict"], strict=False)

        # Load EMA model weights
        ema_model.load_state_dict(state_dict["model_ema_state_dict"], strict=False)

        # Load optimizer state to keep optimizers in sync across GPUs
        try:
            optim.load_state_dict(state_dict["optimizer_state_dict"])
        except ValueError:
            print(f"Rank {rank}: Failed to load optimizer parameters.")

        # Clean up state_dict after use
        del state_dict

    # Wrap with DistributedDataParallel
    model = DDP(model, device_ids=[rank])
    ema_model = DDP(ema_model, device_ids=[rank])
    # model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    # ema_model = DDP(ema_model, device_ids=[rank], find_unused_parameters=True)

    # Update learning rate for each parameter group
    for param_group in optim.param_groups:
        param_group["lr"] = H.optimizer.learning_rate

    if rank == 0:
        current_lr = optim.param_groups[0]["lr"]
        print(f"Current learning rate at initial: {current_lr}")
        print("*" * 100)

    # Ensure synchronization after loading checkpoint
    dist.barrier()

    betas = get_named_beta_schedule(H.diffusion.noise_schedule, H.diffusion.steps)

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

    if H.diffusion.loss_type == "MSE":
        loss_type = gd.LossType.MSE
    elif H.diffusion.loss_type == "L1":
        loss_type = gd.LossType.L1
    else:
        raise Exception("H.diffusion.loss_type must be in [MSE, L1]")

    diffusion = GaussianDiffusion(
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

    schedule_sampler = create_named_schedule_sampler(
        H.diffusion.schedule_sampler, diffusion
    )  # e.g., "uniform"

    train(
        H,
        model,
        ema_model,
        train_loaders,
        optim,
        diffusion,
        schedule_sampler,
        rank,
        **train_kwargs,
    )


def main(argv):
    H = FLAGS.config
    # Read configuration from command-line arguments or a config file,
    # and assign it to H, which contains all necessary parameters and settings for training.

    ngpus_per_node = torch.cuda.device_count()
    nodes = 1  # Only one physical machine
    world_size = ngpus_per_node * nodes
    print("world_size is:", world_size)

    # Use multiprocessing to launch a process on each GPU
    mp.spawn(main_worker, args=(world_size, H), nprocs=ngpus_per_node, join=True)
    # join=True means the main process will wait for all subprocesses to finish before proceeding.
    # In other words, the main process will exit only after all subprocesses spawned by mp.spawn complete.
    # join=False means the main process will not wait for subprocesses to finish and will continue executing subsequent code.
    # In this case, subprocesses will run in the background.


if __name__ == "__main__":
    app.run(main)
