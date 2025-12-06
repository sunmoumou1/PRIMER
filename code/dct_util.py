
import torch
import torch.nn as nn
import numpy as np
import functools 

class LinearDCT(nn.Linear):
    """
    Implements DCT as a linear layer that can handle 2D fields with unequal height and width.
    :param in_features: Input feature dimension
    :param type: Type of DCT used, such as 'dct', 'idct', etc.
    :param norm: Normalization parameter
    """
    def __init__(self, in_features, type, norm=None, bias=False):
        self.type = type
        self.N = in_features
        self.norm = norm
        super(LinearDCT, self).__init__(in_features, in_features, bias=bias)

    def reset_parameters(self):
        # Initialize the weight matrix as a DCT or IDCT matrix
        I = torch.eye(self.N)
        if self.type == 'dct':
            self.weight.data = dct(I, norm=self.norm).data.t()
        elif self.type == 'idct':
            self.weight.data = idct(I, norm=self.norm).data.t()
        self.weight.requires_grad = False  # Do not update weights


def apply_linear_2d(x, linear_layer_h, linear_layer_w):
    """
    Apply the LinearDCT layer to the last two dimensions for 2D DCT.
    :param x: Input signal, shape (B, C, H, W) or (B, H, W) anything...
    :param linear_layer_h: LinearDCT layer for the height direction
    :param linear_layer_w: LinearDCT layer for the width direction
    :return: DCT-transformed result
    """

    X1 = linear_layer_w(x)  
    X2 = linear_layer_h(X1.transpose(-1, -2).contiguous()) 
    return X2.transpose(-1, -2).contiguous()


def dct(x, norm=None):
    """
    1D DCT
    :param x: Input signal
    :param norm: Normalization option
    :return: DCT transformation result
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.view_as_real(torch.fft.fft(v, dim=1))
    # c is for complicated

    k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    1D inverse DCT
    :param X: Input signal
    :param norm: Normalization option
    :return: Inverse DCT transformation result
    """
    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, N) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    # Note that 't' here stands for temporal
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r
    # This section represents complex number multiplication

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)


class DCTGaussianBlur(nn.Module):
    def __init__(self, img_size, std, inv_snr=0.05):
        super().__init__()
        self.inv_snr = inv_snr
        self.base_H, self.base_W = img_size  # Base image height and width
        self.std_H, self.std_W = std if isinstance(std, tuple) else (std, std)  # Standard deviation for Gaussian blur in H and W directions

        # Initialize DCT (Discrete Cosine Transform) and IDCT (Inverse DCT) layers for the base resolution
        self.dct_h_dict = {
            self.base_H: LinearDCT(self.base_H, 'dct'),
        }
        self.dct_w_dict = {
            self.base_W: LinearDCT(self.base_W, 'dct'),
        }

        self.idct_h_dict = {
            self.base_H: LinearDCT(self.base_H, 'idct'),
        }
        self.idct_w_dict = {
            self.base_W: LinearDCT(self.base_W, 'idct'),
        }

    def gaussian_quadrant(self, shape, standards):
        # Generate a 2D Gaussian kernel of given shape using specified standard deviations along each axis.
        # The Gaussian is constructed using the product of exponentials along height and width directions.
        return torch.from_numpy(
            functools.reduce(
                np.multiply,
                (
                    np.exp(-(dx**2) / (2 * sd**2))
                    for sd, dx in zip(standards, np.indices(shape)) # 	Back to an array of shape (2, H, W), where the first channel is the row index i for each point and the second channel is the column index j.
                ),
            )
        )

    def select_dct_layers(self, x_shape, device):
        """
        Select the appropriate DCT and IDCT layers based on the shape of the input tensor.

        Args:
            x_shape: The shape of the input tensor (typically [B, C, H, W])

        Returns:
            Tuple of (dct_h, dct_w, idct_h, idct_w): the DCT and IDCT modules for height and width

        Raises:
            ValueError if the input spatial size is not supported
        """
        H, W = x_shape[-2], x_shape[-1]
        dct_h = self.dct_h_dict.get(H)
        dct_w = self.dct_w_dict.get(W)
        idct_h = self.idct_h_dict.get(H)
        idct_w = self.idct_w_dict.get(W)

        # Move the selected layers to the same device as the input tensor
        dct_h = dct_h.to(device)
        dct_w = dct_w.to(device)
        idct_h = idct_h.to(device)
        idct_w = idct_w.to(device)

        if dct_h is None or dct_w is None or idct_h is None or idct_w is None:
            raise ValueError(f"Unsupported input shape {x_shape[-2:]}; expected predefined (H, W).")

        return dct_h, dct_w, idct_h, idct_w

    def generate_gaussian(self, H, W, device):
        """
        Dynamically generate a Gaussian kernel for given spatial dimensions.

        The standard deviations are adjusted based on the ratio between the
        current and base image resolutions to maintain scale-invariance.

        Args:
            H: Target image height
            W: Target image width
            device: Device to allocate the tensor

        Returns:
            gaussian: Gaussian filter in DCT domain
            gaussian_conj: Conjugate of the Gaussian (used in deblurring)
        """
        scale_factor_H = H / self.base_H
        scale_factor_W = W / self.base_W

        # Rescale the standard deviations accordingly
        adjusted_std_H = self.std_H * scale_factor_H
        adjusted_std_W = self.std_W * scale_factor_W

        # Frequency domain standard deviation approximation (as used in DCT domain filtering)
        gaussian = self.gaussian_quadrant(
            [H, W], [H / (np.pi * adjusted_std_H), W / (np.pi * adjusted_std_W)]
        ).float()

        gaussian_conj = torch.conj(gaussian)  # Complex conjugate used for inverse filtering

        return gaussian.to(device), gaussian_conj.to(device)

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x):
        """
        Apply DCT-based Gaussian blur to the input tensor.

        The blur is performed in the DCT domain by element-wise multiplication with a Gaussian kernel.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Blurred tensor of the same shape
        """
        device = x.device
        dct_h, dct_w, idct_h, idct_w = self.select_dct_layers(x.shape, device)
        H, W = x.shape[-2], x.shape[-1]

        gaussian, _ = self.generate_gaussian(H, W, device)

        # Apply DCT transform in both spatial dimensions
        x = apply_linear_2d(x, dct_h, dct_w)
        x = x * gaussian.to(x.dtype)  # Filter in frequency domain
        x = apply_linear_2d(x, idct_h, idct_w)  # Transform back with IDCT
        return x

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def undo_wiener(self, x):
        """
        Apply inverse filtering using Wiener deconvolution to approximately reverse the blur effect.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Deblurred tensor using Wiener inverse filtering in the DCT domain
        """

        device = x.device
        dct_h, dct_w, idct_h, idct_w = self.select_dct_layers(x.shape, device)
        H, W = x.shape[-2], x.shape[-1]

        gaussian, gaussian_conj = self.generate_gaussian(H, W, device)

        # Apply Wiener deconvolution: G* / (G*G + inv_snr^2), where G is the Gaussian kernel
        x = apply_linear_2d(x, dct_h, dct_w)
        x = (
            x
            * gaussian_conj.to(x.dtype)
            / (gaussian.to(x.dtype) * gaussian_conj.to(x.dtype) + self.inv_snr**2)
        )
        x = apply_linear_2d(x, idct_h, idct_w)
        return x


        