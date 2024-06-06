import os
import torch
from PIL import Image
import numpy as np

from model import GaussianModel
from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians

def render(gaussian, B_SIZE: int = 16, i: int = 0):
    (xys, depths, radii, conics,
        compensation, num_tiles_hit, cov3d) = project_gaussians(
            gaussian.means[i:i+1], gaussian.scales[i:i+1], 1, (gaussian.quats/gaussian.quats.norm(dim=-1, keepdim=True))[i:i+1],
            gaussian.viewmat, gaussian.focal, gaussian.focal, gaussian.W/2, gaussian.H/2, gaussian.H, gaussian.W, B_SIZE)
    
    out_img = rasterize_gaussians(
        xys, depths, radii, conics,
        num_tiles_hit, torch.sigmoid(gaussian.rgbs[i:i+1]), torch.sigmoid(gaussian.opacities[i:i+1]),
        gaussian.H, gaussian.W, B_SIZE, gaussian.background
        )[..., :3]
    
    out_img = (out_img.detach().cpu().numpy() * 255).astype(np.uint8)
    out_img = Image.fromarray(out_img)
    out_dir = os.path.join(os.getcwd(), "renders")
    os.makedirs(out_dir, exist_ok=True)
    out_img.save(f"{out_dir}/render_{i}.png")

def main():
    gaussian = GaussianModel(render=True)
    for i in range(16):
        render(gaussian, i=i)


if __name__ == "__main__":
    main()