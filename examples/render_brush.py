import os
import time
import math
import numpy as np
from pathlib import Path
from typing import Optional
import torch
from torch import optim
import tyro
from PIL import Image
from torchvision.utils import save_image

from loss_utils import l1_loss, l2_loss, ssim
from model import GaussianModel
from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians
from add_gaussian import add_gaussian, BrushModel


def render_brush(old_gaussian):

    # Load MLP
    brush = BrushModel(100)
    brush = brush.to('cuda')
    model_path = os.path.join(os.getcwd(), "renders")
    model_path = os.path.join(model_path, "brush.pth")
    brush.load_state_dict(torch.load(model_path))

    gaussian = GaussianModel(render=True)
    gaussian = brush(old_gaussian, gaussian)

    # project_gaussians
    (xys, depths, radii, conics,
        compensation, num_tiles_hit, cov3d) = project_gaussians(
            gaussian.means, gaussian.scales, 1, gaussian.quats/gaussian.quats.norm(dim=-1, keepdim=True),
            gaussian.viewmat, gaussian.focal, gaussian.focal, gaussian.W/2, gaussian.H/2, gaussian.H, gaussian.W, 16)

    # rasterize_gaussians
    out_img = rasterize_gaussians(
        xys, depths, radii, conics,
        num_tiles_hit, torch.sigmoid(gaussian.rgbs), torch.sigmoid(gaussian.opacities),
        gaussian.H, gaussian.W, 16, gaussian.background
        )[..., :3]
    
    out_img = (out_img.detach().cpu().numpy() * 255).astype(np.uint8)
    out_img = Image.fromarray(out_img)
    out_dir = os.path.join(os.getcwd(), "renders")
    os.makedirs(out_dir, exist_ok=True)
    out_img.save(f"{out_dir}/render_brush.png")

def main():
    old_gaussian = GaussianModel(render=True)
    render_brush(old_gaussian)

if __name__ == "__main__":
    main()