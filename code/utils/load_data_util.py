import torch # type: ignore
from torch.utils.data import Dataset, DataLoader, Subset, DistributedSampler # type: ignore
import numpy as np
import os
import random
from abc import ABC, abstractmethod
from scipy.ndimage import zoom

class BasePrecipitationDataset(ABC):
    def __init__(
        self,
        root_dir=None,
        min_val=None,
        max_val=None,
        mean=None,
        std=None,
        normalization=None, # minmax or standard
        fixed_length=None,
        sample_filter_fn=None,
        clip_min=None,
        clip_max=None,
        expected_img_size=None,
    ):
        self.data_dir = root_dir

        # Get all .npy files in the directory
        self.data_files = [
            os.path.join(self.data_dir, file)
            for file in os.listdir(self.data_dir)
            if file.endswith(".npy")
        ]

        if not self.data_files:
            raise ValueError(f"No .npy files found in directory: {self.data_dir}")

        self.min_val = min_val
        self.max_val = max_val
        self.normalization = normalization
        self.delta = (
            self.max_val - self.min_val
            if self.normalization == "minmax"
            and self.min_val is not None
            and self.max_val is not None
            else None
        )

        self.mean = mean
        self.std = std
        self.fixed_length = fixed_length
        self.sample_filter_fn = sample_filter_fn

        # Expected shape of the data
        self.expected_shape = expected_img_size
        
        self.clip_min = clip_min # only used in standard
        self.clip_max = clip_max


    def __len__(self):
        return self.fixed_length

    @abstractmethod
    def __getitem__(self, idx):
        """
        Subclasses must implement this method
        """
        pass

    def apply_normalization(self, data):
        if self.normalization == "minmax":
            if self.delta is None:
                raise ValueError(
                    "min_val and max_val must be provided for minmax normalization."
                )
            data = (data - self.min_val) / self.delta
            data = (2 * data) - 1
            if not (np.all(data >= -1) and np.all(data <= 1)):
                raise ValueError(f"The normalized data is out of range [-1, 1]")
            if np.isnan(data).any():
                raise ValueError(f"The normalized data involves NaN")
            
        elif self.normalization == "standard":
            if self.mean is None or self.std is None:
                raise ValueError(
                    "mean and std must be provided for standard normalization."
                )
            data = (data - self.mean) / self.std
            data = np.clip(data, self.clip_min, self.clip_max)
            if np.isnan(data).any():
                raise ValueError(f"There are NaN values")
        else:
            raise ValueError(
                "Unsupported normalization type. Choose 'minmax' or 'standard'."
            )
        return data

    def apply_denormalization(self, data):

        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        if self.normalization == "minmax":
            if self.delta is None:
                raise ValueError(
                    "min_val and max_val must be provided for minmax normalization."
                )
            data = (data + 1) / 2
            # Clip the values to the range [0, 1]
            data = np.clip(data, 0, 1)
            data = data * self.delta + self.min_val
            
        elif self.normalization == "standard":
            if self.mean is None or self.std is None:
                raise ValueError(
                    "mean and std must be provided for standard normalization."
                )
            data = np.clip(data, self.clip_min, self.clip_max)
            data = data * self.std + self.mean
        else:
            raise ValueError(
                "Unsupported normalization type. Choose 'minmax' or 'standard'."
            )
        return data


class ERA5Dataset(BasePrecipitationDataset):
    def __getitem__(self, idx):
        max_tries = 3
        for attempt in range(max_tries):
            file_path = random.choice(self.data_files)
            data = np.load(file_path)

            # if data.shape != self.expected_shape:
            #     if isinstance(data, np.ndarray):
            #         zoom_factors = [n / o for n, o in zip(self.expected_shape, data.shape)]
            #         data = zoom(data, zoom=zoom_factors, order=0)
            #     else:
            #         raise TypeError("data must be a numpy.ndarray")                

            if self.sample_filter_fn != None:
                if not self.sample_filter_fn(data):
                    continue

            data = self.apply_normalization(data)
            # Generate a mask matrix: 1 for NaN and 0 for non-nan
            mask = np.isnan(data).astype(np.int32)
        
            return (
                torch.tensor(data, dtype=torch.float32),
                torch.tensor([1, 0, 0], dtype=torch.float32),
                torch.tensor(mask, dtype=torch.int32)
            )

        data = self.apply_normalization(data)
        mask = np.isnan(data).astype(np.int32)
        return (
            torch.tensor(data, dtype=torch.float32),
            torch.tensor([1, 0, 0], dtype=torch.float32),
            torch.tensor(mask, dtype=torch.int32)
        )

class IMERGDataset(BasePrecipitationDataset):
    def __getitem__(self, idx):
        max_tries = 3
        for attempt in range(max_tries):
            file_path = random.choice(self.data_files)
            data = np.load(file_path)

            if data.shape != self.expected_shape:
                raise ValueError(
                    f"data size mismatch, expected: {self.expected_shape}, actual: {data.shape}"
                )

            if self.sample_filter_fn != None:
                if not self.sample_filter_fn(data):
                    continue

            data = self.apply_normalization(data)
            mask = np.isnan(data).astype(np.int32)
        
            return (
                torch.tensor(data, dtype=torch.float32),
                torch.tensor([0, 1, 0], dtype=torch.float32),
                torch.tensor(mask, dtype=torch.int32)
            )

        data = self.apply_normalization(data)
        mask = np.isnan(data).astype(np.int32)
        return (
            torch.tensor(data, dtype=torch.float32),
            torch.tensor([0, 1, 0], dtype=torch.float32),
            torch.tensor(mask, dtype=torch.int32)
        )

class gaugeDataset(BasePrecipitationDataset):
    def __getitem__(self, idx):
        file_path = random.choice(self.data_files)
        data = np.load(file_path)
        
        channel1, channel2 = data[0], data[1]
        channel1 = channel1[np.newaxis, ...]
        channel2 = channel2[np.newaxis, ...]

        if channel1.shape != self.expected_shape:
            raise ValueError(
                f"data size mismatch, expected: {self.expected_shape}, actual: {data.shape}"
            )
            
        channel1 = self.apply_normalization(channel1)
        mask = np.where(channel2 >= 1, 0, 1) # Only grid points with AWS(Automatic Weather Stations) fusion are selected for training
        
        return (
            torch.tensor(channel1, dtype=torch.float32),
            torch.tensor([0, 0, 1], dtype=torch.float32),
            torch.tensor(mask, dtype=torch.int32)
        )


def train_val_split(dataset, train_val_ratio):
    """Split the training and validation sets"""
    indices = list(range(len(dataset)))
    split_index = int(len(dataset) * train_val_ratio)
    train_indices, val_indices = indices[:split_index], indices[split_index:]
    train_dataset, val_dataset = Subset(dataset, train_indices), Subset(
        dataset, val_indices
    )
    return train_dataset, val_dataset


# The following get_data_loader is used for training the example discrete sparse gauge observations!!!!!!!!!!!!!!!!!!!!!!!
# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------

def get_data_loader(
    H,
    drop_last=True,
    train_val_split_ratio=1,
    shuffle=True,
    rank=0,
    world_size=2,
    return_val=False,
):

    train_loaders = []
    val_loaders = []

    # Initialize the dataset
    dataset = gaugeDataset(
        root_dir=H.data.root_dir,
        min_val=H.data.min_val,
        max_val=H.data.max_val,
        mean=H.data.mean,
        std=H.data.std,
        normalization=H.data.normalization,
        fixed_length=H.data.fixed_length,
        sample_filter_fn=H.data.sample_filter_fn,
        clip_min=H.data.clip_min,
        clip_max=H.data.clip_max,
        expected_img_size=H.data.expected_img_size,
    )

    # Training set and validation set split
    train_dataset, val_dataset = train_val_split(dataset, train_val_split_ratio)

    if rank == 0:
        print(f"Dataset - Training set Number of samples: {len(train_dataset)}")
        print(f"Dataset - Validation set Number of samples: {len(val_dataset)}")
        print("*" * 20)

    # Initialize the distributed sampler
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=shuffle
    )
    val_sampler = (
        DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        if return_val
        else None
    )

    train_loaders.append(
        DataLoader(
            train_dataset,
            batch_size=H.train.batch_size,
            sampler=train_sampler,
            drop_last=drop_last,
            num_workers=H.data.num_workers,
        )
    )

    if return_val:
        val_loaders.append(
            DataLoader(
                val_dataset,
                batch_size=H.train.batch_size,
                sampler=val_sampler,
                drop_last=drop_last,
                num_workers=1,
            )
        )

    return train_loaders, val_loaders


# The following get_data_loader is used for training multi-sources precipitation records !!!!!!!!!!!!!!!!!!!!!!!
# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------

# def get_data_loader(
#     H,
#     drop_last=True,
#     train_val_split_ratio=1,
#     shuffle=True,
#     rank=0,
#     world_size=2,
#     return_val=False,
# ):
#     dataset_map = {
#         "ERA5Dataset": ERA5Dataset,
#         "IMERGDataset": IMERGDataset,
#         "gaugeDataset": gaugeDataset,
#     }

#     train_loaders = []
#     val_loaders = []

#     root_dirs = H.data.root_dir
    
#     for i, root_dir in enumerate(root_dirs):
#         if i == 0:
#             dataset_cls_name = "ERA5Dataset"
#         elif i == 1:
#             dataset_cls_name = "IMERGDataset"
#         else:
#             dataset_cls_name = "gaugeDataset"
            
#         dataset_cls = dataset_map[dataset_cls_name]

#         dataset = dataset_cls(
#             root_dir=root_dir,
#             min_val=H.data.min_val,
#             max_val=H.data.max_val,
#             mean=H.data.mean,
#             std=H.data.std,
#             normalization=H.data.normalization,
#             fixed_length=H.data.fixed_length,
#             sample_filter_fn=H.data.sample_filter_fn,
#             clip_min=H.data.clip_min,
#             clip_max=H.data.clip_max,
#             expected_img_size=H.data.expected_img_size,
#         )

#         train_dataset, val_dataset = train_val_split(dataset, train_val_split_ratio)

#         if rank == 0:
#             print(dataset_cls_name, '------------------------------------------')
#             print(f"Dataset - Training set Number of samples: {len(train_dataset)}")
#             print(f"Dataset - Validation set Number of samples: {len(val_dataset)}")
#             print("*" * 20)

#         train_sampler = DistributedSampler(
#             train_dataset, num_replicas=world_size, rank=rank, shuffle=shuffle
#         )
#         val_sampler = (
#             DistributedSampler(
#                 val_dataset, num_replicas=world_size, rank=rank, shuffle=False
#             )
#             if return_val
#             else None
#         )

#         train_loaders.append(
#             DataLoader(
#                 train_dataset,
#                 batch_size=H.train.batch_size,
#                 sampler=train_sampler,
#                 drop_last=drop_last,
#                 num_workers=H.data.num_workers,
#             )
#         )

#         if return_val:
#             val_loaders.append(
#                 DataLoader(
#                     val_dataset,
#                     batch_size=H.train.batch_size,
#                     sampler=val_sampler,
#                     drop_last=drop_last,
#                     num_workers=1,
#                 )
#             )

#     return train_loaders, val_loaders
