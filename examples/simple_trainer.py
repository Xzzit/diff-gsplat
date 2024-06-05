import math
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tyro
from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians
from PIL import Image
from torch import Tensor, optim
from plyfile import PlyData, PlyElement

from loss_utils import l1_loss, l2_loss, ssim


class SimpleTrainer:
    """Trains random gaussians to fit an image."""

    def __init__(self, gt_image: Tensor, num_points: int = 2000):
        self.device = torch.device("cuda:0")
        self.gt_image = gt_image.to(device=self.device)
        self.num_points = num_points

        fov_x = math.pi / 2.0
        self.H, self.W = gt_image.shape[0], gt_image.shape[1]
        self.focal = 0.5 * float(self.W) / math.tan(0.5 * fov_x)
        self.img_size = torch.tensor([self.W, self.H, 1], device=self.device)

        self._init_gaussians()

    def _init_gaussians(self):
        """Random gaussians"""
        self.means = 2 * (torch.rand(self.num_points, 3, device=self.device) - 0.5)

        self.scales = torch.rand(self.num_points, 3, device=self.device)
        
        channel = 3
        self.rgbs = torch.rand(self.num_points, channel, device=self.device)

        u = torch.rand(self.num_points, 1, device=self.device)
        v = torch.rand(self.num_points, 1, device=self.device)
        w = torch.rand(self.num_points, 1, device=self.device)
        self.quats = torch.cat([
            torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
            torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
            torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
            torch.sqrt(u) * torch.cos(2.0 * math.pi * w)
            ], -1)
        
        self.opacities = torch.ones((self.num_points, 1), device=self.device)

        self.viewmat = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 8.0],
            [0.0, 0.0, 0.0, 1.0]
            ], device=self.device)
        
        self.background = torch.zeros(3, device=self.device)

        self.means.requires_grad = True
        self.scales.requires_grad = True
        self.quats.requires_grad = True
        self.rgbs.requires_grad = True
        self.opacities.requires_grad = True
        self.viewmat.requires_grad = False

    def train(self, iterations: int = 1000, lr: float = 0.01,
        save_imgs: bool = False, B_SIZE: int = 16):
        optimizer = optim.Adam(
            [self.rgbs, self.means, self.scales, self.opacities, self.quats], lr)
        frames = []
        times = [0] * 3  # project, rasterize, backward
        for iter in range(iterations):
            # project_gaussians
            start = time.time()
            (xys, depths, radii, conics,
             compensation, num_tiles_hit, cov3d) = project_gaussians(
                 self.means, self.scales, 1, self.quats/self.quats.norm(dim=-1, keepdim=True),
                 self.viewmat, self.focal, self.focal, self.W/2, self.H/2, self.H, self.W, B_SIZE)
            torch.cuda.synchronize()
            times[0] += time.time() - start

            # rasterize_gaussians
            start = time.time()
            out_img = rasterize_gaussians(
                xys, depths, radii, conics,
                num_tiles_hit, torch.sigmoid(self.rgbs), torch.sigmoid(self.opacities),
                self.H, self.W, B_SIZE, self.background
                )[..., :3]
            torch.cuda.synchronize()
            times[1] += time.time() - start

            ratio = 0.1
            loss = ratio * ssim(out_img, self.gt_image) +\
            (1 - ratio) * l1_loss(out_img, self.gt_image) +\
            l2_loss(out_img, self.gt_image)

            optimizer.zero_grad()
            start = time.time()
            loss.backward()
            torch.cuda.synchronize()
            times[2] += time.time() - start
            optimizer.step()
            print(f"Iteration {iter + 1}/{iterations}, Loss: {loss.item()}")

            if save_imgs and iter % 5 == 0:
                frames.append((out_img.detach().cpu().numpy() * 255).astype(np.uint8))
        
        with torch.no_grad():
            self.save_ply()

            if save_imgs:
                # save them as a gif with PIL
                frames = [Image.fromarray(frame) for frame in frames]
                out_dir = os.path.join(os.getcwd(), "renders")
                os.makedirs(out_dir, exist_ok=True)
                frames[0].save(f"{out_dir}/training.gif",
                    save_all=True, append_images=frames[1:],
                    optimize=False, duration=5, loop=0)
        
        print(f"Total(s):\nProject: {times[0]:.3f}, Rasterize: {times[1]:.3f}, Backward: {times[2]:.3f}")
        print(f"Per step(s):\n\
Project: {times[0]/iterations:.5f},\
Rasterize: {times[1]/iterations:.5f},\
Backward: {times[2]/iterations:.5f}")

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z']
        # All channels except the 3 DC
        for i in range(self.rgbs.shape[1]):
            l.append('color{}'.format(i))
        l.append('opacity')
        for i in range(self.scales.shape[1]):
            l.append('scale{}'.format(i))
        for i in range(self.quats.shape[1]):
            l.append('rotation{}'.format(i))
        return l

    def save_ply(self):
        xyz = self.means.detach().cpu().numpy()
        color = self.rgbs.detach().cpu().numpy()
        opacity = self.opacities.detach().cpu().numpy()
        scale = self.scales.detach().cpu().numpy()
        rotation = self.quats.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, color, opacity, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')

        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(current_dir, 'renders')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'points.ply')
        PlyData([el]).write(output_file)
        print(f"Saved ply file to {output_file}")

    def load_ply(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        ply_file_path = os.path.join(current_dir, 'renders', 'points.ply')
        plydata = PlyData.read(ply_file_path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        
        color_name = [p.name for p in plydata.elements[0].properties if p.name.startswith("name")]
        color = np.zeros((xyz.shape[0], len(color_name)))
        for idx, attr_name in enumerate(color_name):
            color[:, idx] = np.asarray(plydata.elements[0][attr_name])

        opacity = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rotation")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self.means = torch.tensor(xyz, dtype=torch.float, device="cuda")
        self.rgbs = torch.tensor(color, dtype=torch.float, device="cuda").contiguous()
        self.opacities = torch.tensor(opacity, dtype=torch.float, device="cuda")
        self.scales = torch.tensor(scales, dtype=torch.float, device="cuda")
        self.quats = torch.tensor(rots, dtype=torch.float, device="cuda")

        print(f'{ply_file_path} loaded successfully!')


def image_path_to_tensor(image_path: Path):
    import torchvision.transforms as transforms

    img = Image.open(image_path)
    transform = transforms.ToTensor()
    img_tensor = transform(img).permute(1, 2, 0)[..., :3]
    return img_tensor


def main(height: int = 256, width: int = 256, num_points: int = 100000,
         save_imgs: bool = True, img_path: Optional[Path] = None, 
         iterations: int = 1000, lr: float = 0.01) -> None:
    
    if img_path:
        gt_image = image_path_to_tensor(img_path)
    else:
        gt_image = torch.ones((height, width, 3)) * 1.0
        # make top left and bottom right red, blue
        gt_image[: height // 2, : width // 2, :] = torch.tensor([1.0, 0.0, 0.0])
        gt_image[height // 2 :, width // 2 :, :] = torch.tensor([0.0, 0.0, 1.0])
    
    trainer = SimpleTrainer(gt_image=gt_image, num_points=num_points)
    trainer.train(iterations=iterations, lr=lr, save_imgs=save_imgs)


if __name__ == "__main__":
    tyro.cli(main)
