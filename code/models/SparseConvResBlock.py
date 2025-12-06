import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import calculate_gain
from einops import rearrange, repeat # type: ignore
import math
from .helper_function import *

# torchsparse
try:
    import torchsparse # type: ignore
    from torchsparse import nn as spnn # type: ignore
    from torchsparse import SparseTensor # type: ignore
    TORCHSPARSE_AVAILABLE = True
except Exception:
    TORCHSPARSE_AVAILABLE = False

class SparseConvResBlock(nn.Module):

    def __init__(
        self,
        img_size,
        nf,
        kernel_size=7,
        mult=2, # This is used to scale the number of neurons in the middle hidden layer of a fully connected network
        skip_dim=None,
        time_emb_dim=None,
        epsilon=1e-5,
        z_dim=None,
        depthwise=True,
        backend="torchsparse", 
    ):
        super().__init__()
        self.backend = backend

        if self.backend == "torchsparse":
            assert TORCHSPARSE_AVAILABLE, "torchsparse backend is not detected."
            block = TorchsparseResBlock
        else:
            raise Exception("Unrecognised backend.")

        self.block = block(
            img_size,
            nf,
            kernel_size=kernel_size,
            mult=mult,
            skip_dim=skip_dim,
            time_emb_dim=time_emb_dim,
            epsilon=epsilon,
            z_dim=z_dim,
            depthwise=depthwise,
        )

    def forward(self, x, t=None, skip=None, z=None, norm=None):
        if isinstance(x, torch.Tensor) and len(x.shape) == 4:
            # If image shape passed in 4, then use more efficient dense convolution
            return self.block.dense_forward(x, t=t, skip=skip, z=z, norm=norm)
        else:
            return self.block(x, t=t, skip=skip, z=z, norm=norm)


def ts_add(a, b):
    if isinstance(b, SparseTensor):
        feats = a.feats + b.feats
    else:
        feats = a.feats + b
    out = SparseTensor(coords=a.coords, feats=feats, stride=a.stride)
    out.cmaps = a.cmaps
    out.kmaps = a.kmaps
    return out


def ts_div(a, b):
    if isinstance(b, SparseTensor):
        feats = a.feats / b.feats
    else:
        feats = a.feats / b
    out = SparseTensor(coords=a.coords, feats=feats, stride=a.stride)
    out.cmaps = a.cmaps
    out.kmaps = a.kmaps
    return out


class TorchsparseResBlock(nn.Module):
    '''
    Note in particular that this class does not depend on the size of the data being processed, so there is no need to enter img_size 
    But why did we pass img_size anyway? This is because in get_torch_kernel we need a reference image size in order to confirm the kernel size
    '''
    def __init__(
        self,
        img_size,
        embed_dim,
        kernel_size=7,
        mult=2, 
        skip_dim=None,
        time_emb_dim=None,
        epsilon=1e-5,
        z_dim=None,
        depthwise=True,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.epsilon = epsilon
        self.embed_dim = embed_dim
        self.groups = embed_dim if depthwise else 1
        self.img_size = img_size

        if skip_dim is not None:
            self.skip_linear = nn.Linear(embed_dim + skip_dim, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.conv = spnn.Conv3d(
            embed_dim,
            embed_dim,
            kernel_size=(1, kernel_size, kernel_size),
            depthwise=depthwise,
            bias=False,
        ) 

        self._custom_kaiming_uniform_(self.conv.kernel, a=math.sqrt(5))

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mult),
            nn.GELU(),
            nn.Linear(embed_dim * mult, embed_dim),
        )

        self.time_mlp1, self.time_mlp2, self.z_mlp1, self.z_mlp2 = (
            None,
            None,
            None,
            None,
        )
        if time_emb_dim is not None:
            self.time_mlp1 = nn.Sequential(
                nn.GELU(), nn.Linear(time_emb_dim, embed_dim * 2)
            )
            self.time_mlp2 = nn.Sequential(
                nn.GELU(), nn.Linear(time_emb_dim, embed_dim * 2)
            )
        if z_dim is not None:
            self.z_mlp1 = nn.Sequential(
                nn.Linear(z_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )
            self.z_mlp2 = nn.Sequential(
                nn.Linear(z_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )

    def forward(self, x, t=None, skip=None, z=None, norm=None):
        assert isinstance(x, torchsparse.SparseTensor)

        # Skip connection
        if skip is not None:
            feats = torch.cat((x.feats, skip.feats), dim=-1)
            feats = self.skip_linear(feats)
            x = convert_to_backend_form_like(
                feats, x, backend="torchsparse", rearrange_x=False
            )

        h = x
        if t is not None or z is not None:
            h = self.modulate(
                h, t=t, z=z, norm=self.norm1, t_mlp=self.time_mlp1, z_mlp=self.z_mlp1
            )

        h = self.conv(h)
        h = ts_div(h, norm)
        x = ts_add(x, h)

        if t is not None or z is not None:
            h = self.modulate(
                x, t=t, z=z, norm=self.norm2, t_mlp=self.time_mlp2, z_mlp=self.z_mlp2
            )
        x = ts_add(x, self.mlp(h.feats))

        return x

    def _custom_kaiming_uniform_(self, tensor, a=0, nonlinearity="leaky_relu"):
        '''
        a: Negative slope parameter used for Leaky ReLU
        '''
        fan = self.embed_dim * (self.kernel_size**2)
        # fan denotes the number of elements in the weight tensor that participate in the input
        gain = calculate_gain(nonlinearity, a)
        std = gain / math.sqrt(fan)
        bound = math.sqrt(4.0) * std  # Calculate uniform bounds from standard deviation
        with torch.no_grad():
            return tensor.uniform_(-bound, bound)

    def modulate(self, h, t=None, z=None, norm=None, t_mlp=None, z_mlp=None):
        if isinstance(h, torchsparse.SparseTensor):
            feats = h.feats
        else:
            feats = h
        feats = norm(feats)

        q_sample = feats.size(0) // t.size(0)
        if t is not None:
            t = t_mlp(t)
            t = repeat(t, "b c -> (b l) c", l=q_sample)
            # The shape after repetition is (b * l, c), that is, (b * q_sample, c), which matches the first dimension of feats
            t_scale, t_shift = t.chunk(2, dim=-1)
            feats = feats * (1 + t_scale) + t_shift
        if z is not None:
            z_scale = z_mlp(z)
            z_scale = repeat(z_scale, "b c -> (b l) c", l=q_sample)
            feats = feats * (1 + z_scale)
            
        if isinstance(h, torchsparse.SparseTensor):
            h = convert_to_backend_form_like(
                feats, h, backend="torchsparse", rearrange_x=False
            )
        else:
            h = feats
        return h

    def get_torch_kernel(self, img_size, round_down=True):
        if isinstance(img_size, int):
            img_height, img_width = img_size, img_size
        elif isinstance(img_size, tuple) and len(img_size) == 2:
            img_height, img_width = img_size
        else:
            raise ValueError("img_size must be either an integer or a tuple of two integers.")

        if isinstance(self.img_size, int):
            self_height, self_width = self.img_size, self.img_size
        elif isinstance(self.img_size, tuple) and len(self.img_size) == 2:
            self_height, self_width = self.img_size
        else:
            raise ValueError("self.img_size must be either an integer or a tuple of two integers.")

        if (img_height, img_width) != (self_height, self_width):
            ratio_height = img_height / self_height
            ratio_width = img_width / self_width

            ratio = min(ratio_height, ratio_width)

            new_kernel_size = self.kernel_size * ratio

            if round_down:
                new_kernel_size = 2 * round((new_kernel_size - 1) / 2) + 1
            else:
                new_kernel_size = math.floor(new_kernel_size / 2) * 2 + 1

            new_kernel_size = max(new_kernel_size, 3)

            kernel = rearrange(self.conv.kernel, "(h w) i o -> o i w h", h=self.kernel_size)
            kernel = F.interpolate(kernel, size=new_kernel_size, mode="bilinear")
            return kernel
        else:
            return rearrange(self.conv.kernel, "(h w) i o -> o i w h", h=self.kernel_size)

    def dense_forward(self, x, t=None, skip=None, z=None, norm=None):
        assert isinstance(x, torch.Tensor), "Dense forward expects x to be a torch Tensor"
        assert len(x.shape) == 4, "Dense forward expects x to be 4D: (b, c, h, w)"

        # Skip connection
        batch_size, height, width = x.size(0), x.size(2), x.size(3)
        h = rearrange(x, "b c h w -> (b h w) c")
        if skip is not None:
            skip = rearrange(skip, "b c h w -> (b h w) c")
            h = torch.cat((h, skip), dim=-1)
            h = self.skip_linear(h)
        x = h # (b h w) c 

        if t is not None or z is not None:
            h = self.modulate(h, t=t, z=z, norm=self.norm1, t_mlp=self.time_mlp1, z_mlp=self.z_mlp1)
        h = rearrange(h, "(b h w) c -> b c h w", b=batch_size, h=height, w=width)

        # Conv and norm
        kernel = self.get_torch_kernel((height, width))
        h = F.conv2d(h, kernel, padding=kernel.size(-1)//2, groups=self.groups)
        h = h / norm
        h = rearrange(h, "b c h w -> (b h w) c")

        x = x + h  # (b h w) c 

        if t is not None or z is not None:
            h = self.modulate(x, t=t, z=z, norm=self.norm2, t_mlp=self.time_mlp2, z_mlp=self.z_mlp2)
        h = self.mlp(h)

        x = x + h # (b h w) c 

        x = rearrange(x, "(b h w) c -> b c h w", b=batch_size, h=height, w=width)

        return x


