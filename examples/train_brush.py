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


def image_path_to_tensor(image_path: Path):
    import torchvision.transforms as transforms

    img = Image.open(image_path)
    transform = transforms.ToTensor()
    img_tensor = transform(img).permute(1, 2, 0)[..., :3]
    return img_tensor

def train(old_gaussian: GaussianModel, gt_image: torch.Tensor,
          iterations: int = 1000, lr: float = 0.01,
          save_imgs: bool = True, B_SIZE: int = 16):
    
    # For mlp
    brush = BrushModel(100)
    brush = brush.to('cuda')
    features = brush(old_gaussian)
    gaussian = GaussianModel(render=True, features=features)
    optimizer = optim.Adam(brush.parameters(), lr)

    # For add_gaussian
    """ Before
    gaussian.means.requires_grad -> False
    gaussian.means.shape -> [1, 3]
    gaussian.means.grad -> None
    """
    # gaussian = add_gaussian(gaussian, 100)
    """ After
    gaussian.means.requires_grad -> True
    gaussian.means.shape -> [101, 3]
    gaussian.means.grad -> None
    """
    # optimizer = optim.Adam(
    #     [gaussian.rgbs, gaussian.means, gaussian.scales, gaussian.opacities, gaussian.quats], lr)
    
    frames = []
    times = [0] * 3  # project, rasterize, backward
    for iter in range(iterations):
        # project_gaussians
        start = time.time()
        (xys, depths, radii, conics,
            compensation, num_tiles_hit, cov3d) = project_gaussians(
                gaussian.means, gaussian.scales, 1, gaussian.quats/gaussian.quats.norm(dim=-1, keepdim=True),
                gaussian.viewmat, gaussian.focal, gaussian.focal, gaussian.W/2, gaussian.H/2, gaussian.H, gaussian.W, B_SIZE)
        torch.cuda.synchronize()
        times[0] += time.time() - start

        # rasterize_gaussians
        start = time.time()
        out_img = rasterize_gaussians(
            xys, depths, radii, conics,
            num_tiles_hit, torch.sigmoid(gaussian.rgbs), torch.sigmoid(gaussian.opacities),
            gaussian.H, gaussian.W, B_SIZE, gaussian.background
            )[..., :3]
        torch.cuda.synchronize()
        times[1] += time.time() - start

        ratio = 0.1
        # loss = ratio * ssim(out_img, gt_image) +\
        # (1 - ratio) * l1_loss(out_img, gt_image) +\
        # l2_loss(out_img, gt_image)
        loss = l1_loss(out_img, gt_image)

        optimizer.zero_grad()   
        start = time.time()
        loss.backward()
        """ (add_gaussian)After
        gaussian.means.grad -> tensor
        gaussian.means.grad.shape -> [101, 3]
        """
        torch.cuda.synchronize()
        times[2] += time.time() - start

        # if gaussian.means.grad[0] is not None:
        #     gaussian.means.grad[0].zero_()
        # if gaussian.scales.grad[0] is not None:
        #     gaussian.scales.grad[0].zero_()
        # if gaussian.quats.grad[0] is not None:
        #     gaussian.quats.grad[0].zero_()
        # if gaussian.rgbs.grad[0] is not None:
        #     gaussian.rgbs.grad[0].zero_()
        # if gaussian.opacities.grad[0] is not None:
        #     gaussian.opacities.grad[0].zero_()

        optimizer.step()
        print(f"Iteration {iter + 1}/{iterations}, Loss: {loss.item()}")

        if save_imgs and iter % 5 == 0:
            frames.append((out_img.detach().cpu().numpy() * 255).astype(np.uint8))
    
    with torch.no_grad():
        gaussian.save_ply(name="points_brush.ply")
        gaussian.save_camera()

        if save_imgs:
            # save them as a gif with PIL
            frames = [Image.fromarray(frame) for frame in frames]
            out_dir = os.path.join(os.getcwd(), "renders")
            os.makedirs(out_dir, exist_ok=True)
            frames[0].save(f"{out_dir}/training_brush.gif",
                save_all=True, append_images=frames[1:],
                optimize=False, duration=5, loop=0)
    
    print(f"Total(s):\nProject: {times[0]:.3f}, Rasterize: {times[1]:.3f}, Backward: {times[2]:.3f}")
    print(f"Per step(s):\nProject: {times[0]/iterations:.5f}, Rasterize: {times[1]/iterations:.5f}, Backward: {times[2]/iterations:.5f}")

def main(height: int = 256, width: int = 256,
         img_path: Optional[Path] = None, 
         iterations: int = 1000, lr: float = 0.01) -> None:
    
    if img_path:
        gt_image = image_path_to_tensor(img_path)
    else:
        gt_image = torch.ones((height, width, 3)) * 1.0
        # make top left and bottom right red, blue
        gt_image[: height // 2, : width // 2, :] = torch.tensor([1.0, 0.0, 0.0])
        gt_image[height // 2 :, width // 2 :, :] = torch.tensor([0.0, 0.0, 1.0])
    
    out_dir = os.path.join(os.getcwd(), "renders")
    os.makedirs(out_dir, exist_ok=True)
    save_image(gt_image.permute(2, 0, 1), f'{out_dir}/gt_brush.png')
    gt_image = gt_image.to('cuda:0')
    
    torch.autograd.set_detect_anomaly(True)
    gaussian = GaussianModel(render=True)
    train(gaussian, gt_image, iterations=iterations, lr=lr)

if __name__ == "__main__":
    tyro.cli(main)