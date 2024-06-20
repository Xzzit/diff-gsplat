import math
import os
import numpy as np
import json
import torch
from plyfile import PlyData, PlyElement


class GaussianModel:
    """Trains random gaussians to fit an image."""

    def __init__(self, num_points: int = 2000, camera: list = {}, render: bool = False, features = None):
        self.device = 'cuda:0'
        self.viewmat = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 8.0],
            [0.0, 0.0, 0.0, 1.0]
            ], device=self.device)
        self.viewmat.requires_grad = False
        self.background = torch.ones(3, device=self.device)
        
        if not render:
            self.num_points = num_points
            self.H = camera[0]
            self.W = camera[1]
            self.focal = camera[2]
            self._init_gaussians()
        else:
            self.load_camera()
            self.load_ply()

        if features:
            self.load_camera()
            self.means = features[0]
            self.scales = features[1]
            self.rgbs = features[2]
            self.quats = features[3]
            self.opacities = features[4]

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

        self.means.requires_grad = True
        self.scales.requires_grad = True
        self.quats.requires_grad = True
        self.rgbs.requires_grad = True
        self.opacities.requires_grad = True

    def save_camera(self):
        out_dir = os.path.join(os.getcwd(), "renders")
        os.makedirs(out_dir, exist_ok=True)

        camera_info = {
            'H': self.H,
            'W': self.W,
            'focal': self.focal
        }
        with open(f"{out_dir}/camera_info.json", 'w') as f:
            json.dump(camera_info, f)

    def load_camera(self):
        in_dir = os.path.join(os.getcwd(), "renders")
        with open(os.path.join(in_dir, 'camera_info.json'), 'r') as f:
            camera_info = json.load(f)
            
        self.H = camera_info['H']
        self.W = camera_info['W']
        self.focal = camera_info['focal']

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

    def save_ply(self, name='points.ply'):
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
        output_file = os.path.join(output_dir, name)
        PlyData([el]).write(output_file)
        print(f"Saved ply file to {output_file}")

    def load_ply(self, name='points.ply'):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        ply_file_path = os.path.join(current_dir, 'renders', name)
        plydata = PlyData.read(ply_file_path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        
        color_name = [p.name for p in plydata.elements[0].properties if p.name.startswith("color")]
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

