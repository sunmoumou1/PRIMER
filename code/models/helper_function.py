import torch
from einops import rearrange, repeat # type: ignore
import math

# torchsparse
try:
    import torchsparse # type: ignore
    from torchsparse import nn as spnn # type: ignore
    from torchsparse import SparseTensor # type: ignore
    TORCHSPARSE_AVAILABLE = True
except Exception:
    TORCHSPARSE_AVAILABLE = False


##############################
###### HELPER FUNCTIONS ######
##############################

def convert_to_backend_form(x, sample_lst, img_size, backend="torchsparse"):

    if backend == "torchsparse":
        sparse_indices = sample_lst_to_sparse_indices(sample_lst, img_size, ndims=3)
        x = torchsparse.SparseTensor(
            coords=sparse_indices, feats=rearrange(x, "b l c -> (b l) c")
        )
    else:
        raise Exception("Unrecognised backend.")

    return x


def convert_to_backend_form_like(
    x,
    backend_tensor,
    backend="torchsparse",
    rearrange_x=True,
):

    if backend == "torchsparse":
        x = torchsparse.SparseTensor(
            coords=backend_tensor.coords,
            feats=rearrange(x, "b l c -> (b l) c") if rearrange_x else x,
            stride=backend_tensor.stride,
        )
        x.cmaps = backend_tensor.cmaps
        x.kmaps = backend_tensor.kmaps
    else:
        raise Exception("Unrecognised backend.")

    return x


def get_features_from_backend_form(x, batchsize, backend="torchsparse"):

    if backend == "torchsparse":
        return rearrange(x.feats, "(b l) c -> b l c", b=batchsize)
    else:
        raise Exception("Unrecognised backend.")


"""
sample_lst is a tensor of shape (B, L)
which can be used to index flattened 2D images.
This functions converts it to a tensor of shape (BxL, 3)
    indices[:,0] is the number of the item in the batch
    indices[:,1] is the number of the item in the y direction
    indices[:,2] is the number of the item in the x direction
"""
def sample_lst_to_sparse_indices(sample_lst, img_size, ndims=3, dtype=torch.int32):

    if isinstance(img_size, int):
        img_height, img_width = img_size, img_size
    elif isinstance(img_size, tuple) and len(img_size) == 2:
        img_height, img_width = img_size
    else:
        raise ValueError("img_size must be either an integer or a tuple of two integers.")

    # number of the item in the batch - (B,)
    batch_idx = torch.arange(
        sample_lst.size(0), device=sample_lst.device, dtype=dtype
    )
    batch_idx = repeat(batch_idx, "b -> b l", l=sample_lst.size(1))

    # pixel number in vertical direction - (B,L)
    sample_lst_h = sample_lst.div(img_width, rounding_mode="trunc").to(dtype)  # 纵向索引
    # pixel number in horizontal direction - (B,L)
    sample_lst_w = (sample_lst % img_width).to(dtype)  # 横向索引

    if ndims == 2:
        indices = torch.stack([batch_idx, sample_lst_h, sample_lst_w], dim=2)
        indices = rearrange(indices, "b l three -> (b l) three")
    else:
        zeros = torch.zeros_like(sample_lst_h)
        indices = torch.stack([zeros, sample_lst_h, sample_lst_w, batch_idx], dim=2)
        indices = rearrange(indices, "b l four -> (b l) four")

    return indices


def get_normalising_conv(kernel_size, backend="torchsparse"):
    if backend == "torchsparse":
        assert TORCHSPARSE_AVAILABLE, "torchsparse backend is not detected."
        weight = torch.ones(kernel_size**2, 1, 1) / (kernel_size**2)
        conv = spnn.Conv3d(1, 1, kernel_size=(1, kernel_size, kernel_size), bias=False)
        conv.kernel.data = weight
        conv.kernel.requires_grad_(False)
    else:
        raise Exception("Unrecognised backend.")

    return conv


def calculate_norm(conv, backend_tensor, backend="torchsparse"):
    if backend == "torchsparse":
        device, dtype = backend_tensor.feats.device, backend_tensor.feats.dtype
        ones = torch.ones(backend_tensor.feats.size(0), 1, device=device, dtype=dtype)
        mask = torchsparse.SparseTensor(
            coords=backend_tensor.coords,
            feats=ones,
            stride=backend_tensor.stride
        )
        mask.cmaps = backend_tensor.cmaps
        mask.kmaps = backend_tensor.kmaps
        
        # 计算 norm
        norm = conv(mask)
        
        # 检查 norm 是否存在为 0 的元素
        if torch.any(norm.feats == 0):
            print("Warning: 'norm' contains elements equal to 0. These will be directly replaced with mask.")
            norm = mask

    else:
        raise Exception("Unrecognized backend.")
    
    return norm