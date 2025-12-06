import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
from einops import rearrange, repeat # type: ignore
from pytorch3d.ops import knn_points, knn_gather # type: ignore 
import math
import warnings

from .helper_function import *
from .UNO import UNO
from .SparseConvResBlock import (
    SparseConvResBlock,
)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class SparseUNet(nn.Module):
    def __init__(
        self,
        channels=1,
        nf=64,
        time_emb_dim=256,
        img_size=128, # Or a tuple
        num_conv_blocks=3,
        knn_neighbours=3,
        uno_res=64, # Or a tuple
        uno_mults=(1, 2, 4, 8),
        z_dim=None,
        out_channels=None, 
        conv_type="conv", # On UNO
        depthwise_sparse=True, # Indicates whether sparse depthwise convolutions are used, acting on SparseConvResBlock
        kernel_size=7,
        backend="torchsparse",
        blocks_per_level=(2, 2, 2, 2), # 作用于UNO
        attn_res=16,
        dropout_res=16,
        dropout=0.1,
        uno_base_nf=64,
    ):
        super().__init__()
        self.backend = backend
        self.img_size = img_size
        self.uno_res = uno_res
        self.knn_neighbours = knn_neighbours
        self.kernel_size = kernel_size

        # Input projection
        self.linear_in = nn.Linear(channels, nf)
        # Output projection
        self.linear_out = nn.Linear(
            nf, out_channels if out_channels is not None else channels
        )
        
        self.z_mlp = nn.Sequential(
            nn.Linear(z_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, 2 * time_emb_dim),
            nn.GELU(),
            nn.Linear(2 * time_emb_dim, 2 * time_emb_dim),
            nn.GELU(),
            nn.Linear(2 * time_emb_dim, time_emb_dim),
        )
        
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Check the type of uno res and handle it accordingly
        if isinstance(uno_res, int):
            uno_coords = torch.stack(
                torch.meshgrid(*[torch.linspace(0, 1, steps=uno_res) for _ in range(2)])
            )
        elif isinstance(uno_res, tuple) and len(uno_res) == 2:
            uno_coords = torch.stack(
                torch.meshgrid(torch.linspace(0, 1, steps=uno_res[0]), torch.linspace(0, 1, steps=uno_res[1]))
            )
        else:
            raise ValueError("uno_res must be an integer or a tuple of two integers.")

        # Permuting the coordinates
        uno_coords = rearrange(uno_coords, "c h w -> () (h w) c")  
        self.register_buffer("uno_coords", uno_coords)
        # Note that the uno_coords are fixed

        self.normalising_conv = get_normalising_conv(
            kernel_size=kernel_size, backend=backend
        )

        self.down_blocks = nn.ModuleList([])
        for _ in range(num_conv_blocks):
            self.down_blocks.append(
                SparseConvResBlock(
                    img_size,
                    nf,
                    kernel_size=kernel_size,
                    mult=2,
                    time_emb_dim=time_emb_dim,
                    z_dim=time_emb_dim,
                    depthwise=depthwise_sparse,
                    backend=backend,
                )
            )

        self.uno_linear_in = nn.Linear(nf, uno_base_nf)
        self.uno_linear_out = nn.Linear(uno_base_nf, nf)

        self.up_blocks = nn.ModuleList([])
        for _ in range(num_conv_blocks):
            self.up_blocks.append(
                SparseConvResBlock(
                    img_size,
                    nf,
                    kernel_size=kernel_size,
                    mult=2,
                    skip_dim=nf,
                    time_emb_dim=time_emb_dim,
                    z_dim=time_emb_dim,
                    depthwise=depthwise_sparse,
                    backend=backend,
                )
            )

        self.uno = UNO(
            uno_base_nf,
            uno_base_nf,
            width=uno_base_nf,
            mults=uno_mults,
            blocks_per_level=blocks_per_level,
            time_emb_dim=time_emb_dim,
            z_dim=time_emb_dim,
            conv_type=conv_type,
            res=uno_res,
            attn_res=attn_res,
            dropout_res=dropout_res,
            dropout=dropout,
        )

    def knn_interpolate_to_grid(self, x, coords):
        # Note that the incoming coords are subsampled, of shape (B,L,2), representing the coordinates of each lattice point in x
        # x is (B,L,C)

        with torch.no_grad():
            _, assign_index, neighbour_coords = knn_points(
                self.uno_coords.repeat(x.size(0), 1, 1),
                coords, 
                K=self.knn_neighbours,
                return_nn=True,
            )

            # neighbour_coords: (B, y_length, K, 2)
            diff = neighbour_coords - self.uno_coords.unsqueeze(
                2
            )

            squared_distance = (diff * diff).sum(dim=-1, keepdim=True)
            weights = 1.0 / torch.clamp(
                squared_distance, min=1e-15
            )  # (B, y_length, K, 1)

        # Inverse square distance weighted mean
        neighbours = knn_gather(x, assign_index)  # (B, y_length, K, C)

        out = (neighbours * weights).sum(2) / weights.sum(2)

        return out.to(x.dtype)

    def forward(self, x, t, z=None, sample_lst=None, coords=None, base_img_size=None):
        '''
            Note that base_img_size is self-contained; there is no definition in Sam's code
        '''
        z = self.z_mlp(z)
        
        # If x is image shaped (4D) then treat it as a dense tensor for better optimisation
        if len(x.shape) == 4:
            if sample_lst is not None:
                warnings.warn(
                    "Ignoring sample_lst: Recieved 4D x and sample_list != None but treating x as a dense Image."
                )
            if coords is not None:
                warnings.warn(
                    "Ignoring coords: Recieved 4D x and coords != None but treating x as a dense Image."
                )
            return self.dense_forward(x, t, z=z)

        img_height, img_width = base_img_size
        # This line goes after self.dense_forward(x, t, z=z)

        assert sample_lst is not None, "In sparse mode sample_lst must be provided"

        if coords is None:
            # Grid coordinates are generated based on the height and width of the image
            coords = torch.stack(
                torch.meshgrid(
                    torch.linspace(0, 1, steps=img_height),
                    torch.linspace(0, 1, steps=img_width)
                )
            ).to(x.device)
            coords = rearrange(coords, "c h w -> () (h w) c")
            coords = repeat(coords, "() ... -> b ...", b=x.size(0))
            coords = torch.gather(
                coords, 1, sample_lst.unsqueeze(2).repeat(1, 1, coords.size(2))
            ).contiguous()
            # Sampling according to sample_lst, the second dimension becomes L, where L is the length of sample_lst (that is, the number of indices sampled). The shape of the final result is (B, L, 2).

        x = self.linear_in(x)
        t = self.time_mlp(t)

        # 1. Down conv blocks
        x = convert_to_backend_form(x, sample_lst, (img_height,img_width), backend=self.backend)
        # Note that the x returned here comes from the following code:
        # sparse_indices = sample_lst_to_sparse_indices(sample_lst, img_size, ndims=3)
        # x = torchsparse.SparseTensor(
        #     coords=sparse_indices, feats=rearrange(x, "b l c -> (b l) c")
        # )

        backend_tensor = x
        norm = calculate_norm(
            self.normalising_conv,
            backend_tensor,
            backend=self.backend,
        ) 
        # This line essentially returns a tensor in the same sparse position as backend_tensor for Norm

        downs = []
        for block in self.down_blocks:
            x = block(x, t=t, z=z, norm=norm)
            downs.append(x)

        # 2. Interpolate to regular grid
        x = get_features_from_backend_form(x, sample_lst.size(0), backend=self.backend)
        # return rearrange(x.feats, "(b l) c -> b l c", b=batchsize)
        x = self.uno_linear_in(x)
        x = self.knn_interpolate_to_grid(x, coords)
        x = rearrange(x, "b (h w) c -> b c h w", h=self.uno_res[0])

        # 3. UNO
        x = self.uno(x, t, z=z)

        # 4. Interpolate back to sparse coordinates
        x = F.grid_sample(x, coords.unsqueeze(2), mode="bilinear", align_corners=False)
        # Note that in the above line, x is b, c, h, w, and coords.unsqueeze(2) is (b, L, 1, 2).
        # If the input shape is (N, C, H, W) and the grid shape is (N, H_out, W_out, 2), then the output shape is (N, C, H_out, W_out).
        x = rearrange(x, "b c l () -> b l c")
        x = self.uno_linear_out(x)
        x = convert_to_backend_form_like(
            x,
            backend_tensor,
            backend=self.backend,
        )

        # 5. Up conv blocks
        for block in self.up_blocks:
            skip = downs.pop()
            x = block(x, t=t, z=z, skip=skip, norm=norm)

        x = get_features_from_backend_form(x, sample_lst.size(0), backend=self.backend)
        # return rearrange(x.feats, "(b l) c -> b l c", b=batchsize)

        x = self.linear_out(x)

        return x

    def get_torch_norm_kernel_size(self, img_size, round_down=True):
        # Determine whether img_size and self.img_size are integers or tuples and treat them separately
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
            # new kernel_size becomes:
            # 1 -> 1, 1.5 -> 1, 2 -> 1 or 3, 2.5 -> 3, 3 -> 3, 3.5 -> 3, 4 -> 3 or 5, 4.5 -> 5, ...
            # where there are multiple options this is determined by round_down

            ratio = min(ratio_height, ratio_width)

            new_kernel_size = self.kernel_size * ratio
            if round_down:
                new_kernel_size = 2 * round((new_kernel_size - 1) / 2) + 1
            else:
                new_kernel_size = math.floor(new_kernel_size / 2) * 2 + 1
            return max(int(new_kernel_size), 3)  # Make sure the return value is an integer
        else:
            return self.kernel_size

    def dense_forward(self, x, t, z=None):
        # Note that z is already processed in the forward method
          
        # Get the height and width of the input image
        height, width = x.size(2), x.size(3)

        # The coordinate grid is generated and transferred to the device where the tensor is input
        coords = torch.stack(
            torch.meshgrid(
                torch.linspace(0, 1, steps=height), 
                torch.linspace(0, 1, steps=width)
            )
        ).to(x.device)
        coords = rearrange(coords, "c h w -> () (h w) c")
        coords = repeat(coords, "() ... -> b ...", b=x.size(0))

        # Apply the trained parameters of self.linear_in to the convolution operation
        x = F.conv2d(
            x, self.linear_in.weight[:, :, None, None], bias=self.linear_in.bias
        )
        # This line uses the trained parameters of self.linear_in to go from B, C, H, W to B, nf, H, W
        t = self.time_mlp(t)

        # NOTE: Normalization to avoid edge artifacts, in fact, such norm also exists in forward function
        mask = torch.ones(
            x.size(0), 1, x.size(2), x.size(3), dtype=x.dtype, device=x.device
        )
        kernel_size = self.get_torch_norm_kernel_size((height, width))
        weight = torch.ones(
            1, 1, kernel_size, kernel_size, dtype=x.dtype, device=x.device
        ) / (self.kernel_size**2)
        norm = F.conv2d(mask, weight, padding=kernel_size // 2)

        # 1. 下采样卷积块
        downs = []
        for block in self.down_blocks:
            x = block(x, t=t, z=z, norm=norm)
            downs.append(x)

        # 2. 插值到常规网格
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.uno_linear_in(x)
        x = self.knn_interpolate_to_grid(x, coords)
        x = rearrange(x, "b (h w) c -> b c h w", h=self.uno_res[0], w=self.uno_res[1])

        # 3. UNO 处理
        x = self.uno(x, t, z=z)

        # 4. 插值回稀疏坐标
        x = F.grid_sample(x, coords.unsqueeze(2), mode="bilinear", align_corners=False)
        x = rearrange(x, "b c (h w) () -> b c h w", h=height, w=width)
        x = F.conv2d(
            x,
            self.uno_linear_out.weight[:, :, None, None],
            bias=self.uno_linear_out.bias,
        )

        # 5. 上采样卷积块
        for block in self.up_blocks:
            skip = downs.pop()
            x = block(x, t=t, z=z, skip=skip, norm=norm)

        x = F.conv2d(
            x, self.linear_out.weight[:, :, None, None], bias=self.linear_out.bias
        )

        return x
