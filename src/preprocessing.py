import torch
import numpy as np
import skimage.filters
import skimage.restoration
import scipy.ndimage as ndimage

class ApplyFrangiVesselness(object):
    def __init__(self, sigmas=(4, 8), alpha=0.5, beta=0.5, black_ridges=True):
        self.sigmas = sigmas
        self.alpha = alpha
        self.beta = beta
        self.black_ridges = black_ridges

    def __call__(self, img_tensor):
        if img_tensor.dim() == 3 and img_tensor.shape[0] == 1:
            image_np = img_tensor.squeeze(0).cpu().numpy()
        else:
            raise ValueError(f"ApplyFrangiVesselness expects a (1, H, W) tensor, but got {img_tensor.shape}")

        vesselness_np = skimage.filters.frangi(
            image_np,
            sigmas=self.sigmas,
            alpha=self.alpha,
            beta=self.beta,
            black_ridges=self.black_ridges
        )

        vesselness_tensor = torch.from_numpy(vesselness_np).float().unsqueeze(0)

        return vesselness_tensor


class ApplyCEDFilter(object):
    def __init__(self, iterations=10, K=0.03, lambda_param=0.05, option=1):
        self.iterations = iterations
        self.K = K
        self.lambda_param = lambda_param
        self.option = option

    def __call__(self, img_tensor):
        if img_tensor.dim() == 3 and img_tensor.shape[0] == 1:
            image_np = img_tensor.squeeze(0).cpu().numpy()
        else:
            raise ValueError(f"ApplyCEDFilter expects a (1, H, W) tensor, but got {img_tensor.shape}")

        image_np = image_np.astype(np.float32)

        diffused_image_np = np.copy(image_np)

        for _ in range(self.iterations):
            d_north = np.zeros_like(image_np)
            d_south = np.zeros_like(image_np)
            d_east = np.zeros_like(image_np)
            d_west = np.zeros_like(image_np)

            d_north[1:, :] = diffused_image_np[:-1, :] - diffused_image_np[1:, :]
            d_south[:-1, :] = diffused_image_np[1:, :] - diffused_image_np[:-1, :]
            d_east[:, 1:] = diffused_image_np[:, :-1] - diffused_image_np[:, 1:]
            d_west[:, :-1] = diffused_image_np[:, 1:] - diffused_image_np[:, :-1]

            cn = np.zeros_like(image_np)
            cs = np.zeros_like(image_np)
            ce = np.zeros_like(image_np)
            cw = np.zeros_like(image_np)

            if self.option == 1:
                cn[1:, :] = np.exp(-(d_north[1:, :] / self.K)**2)
                cs[:-1, :] = np.exp(-(d_south[:-1, :] / self.K)**2)
                ce[:, 1:] = np.exp(-(d_east[:, 1:] / self.K)**2)
                cw[:, :-1] = np.exp(-(d_west[:, :-1] / self.K)**2)
            elif self.option == 2:
                cn[1:, :] = 1.0 / (1.0 + (d_north[1:, :] / self.K)**2)
                cs[:-1, :] = 1.0 / (1.0 + (d_south[:-1, :] / self.K)**2)
                ce[:, 1:] = 1.0 / (1.0 + (d_east[:, 1:] / self.K)**2)
                cw[:, :-1] = 1.0 / (1.0 + (d_west[:, :-1] / self.K)**2)
            else:
                raise ValueError("CED option must be 1 or 2.")
            
            diffused_image_np += self.lambda_param * (
                cn * d_north + cs * d_south + ce * d_east + cw * d_west
            )
            
        ced_output_tensor = torch.from_numpy(diffused_image_np).float().unsqueeze(0)
        return ced_output_tensor


if __name__ == '__main__':
    print("Testing preprocessing transforms...")

    dummy_img_tensor = torch.rand((1, 256, 256)).float() * 1000 - 500

    print(f"Original dummy image tensor shape: {dummy_img_tensor.shape}")

    try:
        frangi_transform = ApplyFrangiVesselness(sigmas=(1, 2, 4), alpha=0.5, beta=0.5, black_ridges=True)
        frangi_output = frangi_transform(dummy_img_tensor)
        print(f"Frangi output tensor shape: {frangi_output.shape}")
        print(f"Frangi output min/max: {frangi_output.min():.2f}/{frangi_output.max():.2f}")
    except TypeError as e:
        print(f"Error testing Frangi transform: {e}")
    except Exception as e:
        print(f"Unexpected error with Frangi transform: {e}")

    try:
        ced_transform = ApplyCEDFilter(iterations=5, K=0.03, lambda_param=0.05, option=1)
        ced_output = ced_transform(dummy_img_tensor)
        print(f"CED output tensor shape: {ced_output.shape}")
        print(f"CED output min/max: {ced_output.min():.2f}/{ced_output.max():.2f}")
    except Exception as e:
        print(f"Error testing CED transform: {e}")