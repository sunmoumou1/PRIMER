from ml_collections import ConfigDict # type: ignore
import numpy as np
import random
import torch # type: ignore


def default_sample_filter_fn(data, prec_threshold_ratio=0.02, skip_probability=0.8):
    """
    You can create your own sample function aiming to discard samples that you dont like

    Args:
        data (numpy.ndarray): Input data array. The input `data` is likely to have been log-transformed.
        prec_threshold_ratio (float): Threshold ratio for the number of precipitation points
                                      relative to the total number of grid points.
        skip_probability (float): Probability of skipping the sample when precipitation ratio is below the threshold.

    Returns:
        bool: Whether to keep the sample (True = keep, False = skip).
    """
    # Count the number of precipitation grid points and compute the ratio
    prec_count = np.count_nonzero(data > -1)
    total_points = data.size  # Total number of elements in the matrix
    prec_ratio = prec_count / total_points

    # If precipitation ratio is below the threshold, skip the sample with a certain probability
    if prec_ratio < prec_threshold_ratio:
        return random.random() >= skip_probability
    return True


def get_config():
    config = ConfigDict()

    config.run = run = ConfigDict()
    run.name = 'PRIMER'  # Used to distinguish experiments in W&B
    run.experiment = 'finetuning_with_gauges'  # Used to name the checkpoint folder, you can change by whatever you like 
    run.wandb_dir = ''
    run.wandb_mode = 'online'
    run.unique_name = None  # Optional tag used in sample generation

    config.data = data = ConfigDict()
    data.name = 'precipitation'
    data.root_dir = "./gauges_npy" # for single source training!!!!!!!!!!!!!!!!!!!!!!!
    # data.root_dir = ["./ERA5_npy", "./IMERG_npy", "./gauges_npy"] # for multi-sources training!!!!!!!!!!!!!!!!!!!!!!!


    # Used to verify input shape in the DataLoader
    data.expected_img_size = (1, 250, 250) # the minimum resolution
    data.channels = 1
    data.normalization = 'standard'
    # Options: 'minmax' or 'standard'
    # If using 'minmax', set mean and std to None.
    # If using 'standard', set max_val and min_val to None.

    data.property_path = './process_gauges_before/property_results.npy'
    data.max_val = np.load(data.property_path, allow_pickle=True).item()[
        'max_val']
    data.min_val = np.load(data.property_path, allow_pickle=True).item()[
        'min_val']
    data.mean = np.load(data.property_path, allow_pickle=True).item()['mean']
    data.std = np.load(data.property_path, allow_pickle=True).item()['std']
    data.clip_min = np.load(data.property_path, allow_pickle=True).item()[
        'clip_min']
    data.clip_max = np.load(data.property_path, allow_pickle=True).item()[
        'clip_max']
    data.sample_filter_fn = None # customize the filtering function whatever you like
    data.fixed_length = 100  # Used to fix the dataset length, change whatever you like
    data.num_workers = 5  # Number of workers in the DataLoader

    config.train = train = ConfigDict()
    train.load_checkpoint = False
    train.checkpoint_num = None  # Resume from this checkpoint step
    train.retrain_epoch = None  # Indicates how many epochs have been completed already
    train.amp = True  # Enable automatic mixed precision training
    train.batch_size = 1
    train.plot_graph_steps = 5 # change whatever you like
    train.ema_update_every = 10
    train.ema_decay = 0.995
    # Whether to reset global_step to 0 when resuming
    train.whether_global_step_zero = False
    train.sources_proportion = [0.2, 0.4, 0.4]

    config.model = model = ConfigDict()
    model.nf = 192
    model.time_emb_dim = model.nf * 2
    model.num_conv_blocks = 10
    model.knn_neighbours = 3
    # Whether to use depthwise sparse convolution in SparseConvResBlock
    model.depthwise_sparse = True
    model.kernel_size = 7
    model.backend = "torchsparse"
    model.uno_res = (128, 128)
    model.uno_base_channels = 200
    model.uno_mults = (1, 2, 4, 8)
    model.uno_blocks_per_level = (2, 2, 2, 2)
    model.uno_attn_resolutions = int(model.uno_res[0] / 4)
    model.uno_dropout_from_resolution = int(model.uno_res[0] / 4)
    model.uno_dropout = 0.1
    model.uno_conv_type = "conv"  # Options: 'conv' or 'spectral'; we recommend 'conv'
    model.z_dim = 3
    model.sigma_small = True  # Try False for alternative behavior

    config.diffusion = diffusion = ConfigDict()
    diffusion.steps = 1000
    diffusion.noise_schedule = 'cosine' # Options: 'linear' or 'cosine' or 'const0.008'
    diffusion.schedule_sampler = 'uniform' # Options: 'uniform' or 'loss-second-moment'
    diffusion.loss_type = 'MSE'  # Options: 'MSE' or 'L1'
    diffusion.gaussian_filter_std = 0.5
    diffusion.model_mean_type = "mollified_epsilon"
    diffusion.mollifier_type = "dct"
    # Note: 'conv' mollifier type is not supported; the original implementation is incomplete.
    # The conv-based mollifier lacks the `.undo_wiener()` method and is not functional.

    config.mc_integral = mc_integral = ConfigDict()
    mc_integral.type = 'uniform'
    mc_integral.q_sample_ratio = 0.2  # Ratio of grid points sampled for MC, you can change it for yourself!!!!!!!!!!!!!!!!!!!!!!!
    # mc_integral.q_sample_ratio = [0.2, 0.2, 0.2]  # Ratio of grid points sampled for MC, you can change it for yourself!!!!!!!!!!!!!!!!!!!!!!!

    config.optimizer = optimizer = ConfigDict()
    optimizer.learning_rate = 1e-5  # Initial learning rate (not decayed)
    optimizer.adam_beta1 = 0.9
    optimizer.adam_beta2 = 0.99
    optimizer.warmup_steps = 1000
    optimizer.gradient_skip = True
    optimizer.gradient_skip_threshold = 100
    optimizer.weight_decay = 4e-6 
    # You can try another values, maybe the model will perform better, but we dont test it thoroughly
    optimizer.decay_interval = 5
    optimizer.minimum_lr = 1e-6

    # Configuration for sample generation (used in `generate_samples`)
    config.generation = generation = ConfigDict()
    generation.sampling_steps = 1000  # Number of steps for sampling
    generation.sample_img_size = (250, 250) # Target resolution for generated samples
    generation.sample_size = 1  # Number of samples to generate
    generation.sample_num = 1  # Number of times to repeat sample_size
    generation.idx = (0, 0, 1) # style

    return config


config = get_config()
