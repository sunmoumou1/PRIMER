
import torch # type: ignore
from ml_collections import ConfigDict # type: ignore
import wandb # type: ignore
import os
import sys
from tqdm import tqdm

# Add the 'utils' directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from load_data_util import get_data_loader


def plot_images(H, x, title=""):
    x = wandb.Image(x, caption=title)
    return {title: x}


def flatten_collection(d, parent_key="", sep="_"):
    """
    The function `flatten_collection` is used to convert a nested dictionary (e.g., a configuration dictionary with multiple levels)
    into a flat dictionary. In the resulting dictionary, nested keys are concatenated into a new single-level key using a specified separator.

    Args:
        d (dict or ConfigDict): The input nested dictionary.
        parent_key (str): The key prefix used during recursion (default is an empty string).
        sep (str): The separator used to join nested keys (default is underscore "_").

    This function recursively traverses all key-value pairs in the dictionary `d`.
    If a value `v` is itself a nested dictionary (e.g., of type ConfigDict), the function recursively calls `flatten_collection`
    and extends the result into the `items` list. If the value is not a nested dictionary, the key-value pair is directly added to `items`.

    Returns:
        dict: A flattened dictionary where all keys from nested structures are concatenated into flat keys using the separator.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, ConfigDict):
            items.extend(flatten_collection(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

# Note the difference between items.extend() and items.append():
# The extend() method adds each element of an iterable (like a list, tuple, or set) individually to the end of the list.
# That means the elements of the iterable are added one-by-one, rather than adding the iterable itself as a single element.

def upscale_sample_indices(
    sample_lst: torch.LongTensor, old_shape: tuple, new_shape: tuple
) -> torch.LongTensor:
    """
    Maps flat indices from a low-resolution grid to a high-resolution grid.

    Args:
        sample_lst (torch.LongTensor): Tensor of shape (B, L), where each element 
            is a flattened index in the range [0, H*W) representing positions in the 
            low-resolution grid.
        old_shape (tuple): Original grid shape as (H, W).
        new_shape (tuple): Target high-resolution grid shape as (H2, W2).

    Returns:
        torch.LongTensor: Tensor of shape (B, L), containing the mapped flattened indices 
            in the high-resolution grid. Each original index is randomly mapped to one 
            of the corresponding locations in the upscaled grid.
    """
    B, L = sample_lst.shape
    H, W = old_shape
    H2, W2 = new_shape

    # Convert flat indices to 2D coordinates (row, column)
    rows = sample_lst // W  # shape: (B, L)
    cols = sample_lst % W   # shape: (B, L)

    # Map original coordinates to regions in the new resolution
    rows_start = (rows.float() * H2 / H) - 1
    rows_end = (rows.float() * H2 / H) + 1
    cols_start = (cols.float() * W2 / W) - 1
    cols_end = (cols.float() * W2 / W) + 1

    # Clamp the start and end positions to stay within valid bounds of the high-res grid
    rs = torch.clamp(torch.ceil(rows_start).long(), min=0, max=H2 - 1).to("cpu")
    re = torch.clamp(torch.floor(rows_end).long(), min=0, max=H2 - 1).to("cpu")
    cs = torch.clamp(torch.ceil(cols_start).long(), min=0, max=W2 - 1).to("cpu")
    ce = torch.clamp(torch.floor(cols_end).long(), min=0, max=W2 - 1).to("cpu")

    # Randomly sample new row and column indices within the mapped region
    rand_rows = torch.rand(B, L).to("cpu")
    rand_cols = torch.rand(B, L).to("cpu")
    new_rows = rs + ((re - rs + 1).clamp(min=1) * rand_rows).floor().long()
    new_cols = cs + ((ce - cs + 1).clamp(min=1) * rand_cols).floor().long()

    # Convert 2D coordinates back to flat indices in the high-resolution grid
    new_indices = new_rows * W2 + new_cols  # shape: (B, L)

    return new_indices