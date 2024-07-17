import torch
import torch.nn as nn
import math
from model import GaussianModel

class BrushModel(nn.Module):
    def __init__(self, num_points: int = 10):
        super(BrushModel, self).__init__()

        self.mlp_means = nn.ModuleList([nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        ) for _ in range(num_points)])

        self.mlp_scales = nn.ModuleList([nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        ) for _ in range(num_points)])

        self.mlp_rgbs = nn.ModuleList([nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        ) for _ in range(num_points)])

        self.mlp_quats = nn.ModuleList([nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        ) for _ in range(num_points)])

        self.mlp_opacities = nn.ModuleList([nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ) for _ in range(num_points)])
    
    def forward(self, old_gaussian: GaussianModel, gaussian: GaussianModel):
        means = torch.cat([mlp(old_gaussian.means) for mlp in self.mlp_means], 0)
        scales = torch.cat([mlp(old_gaussian.scales) for mlp in self.mlp_scales], 0)
        rgbs = torch.cat([mlp(old_gaussian.rgbs) for mlp in self.mlp_rgbs], 0)
        quats = torch.cat([mlp(old_gaussian.quats) for mlp in self.mlp_quats], 0)
        opacities = torch.cat([mlp(old_gaussian.opacities) for mlp in self.mlp_opacities], 0)

        gaussian.means = means
        gaussian.scales = scales
        gaussian.rgbs = rgbs
        gaussian.quats = quats
        gaussian.opacities = opacities

        # gaussian.means = torch.cat([gaussian.means, means], 0)
        # gaussian.scales = torch.cat([gaussian.scales, scales], 0)
        # gaussian.rgbs = torch.cat([gaussian.rgbs, rgbs], 0)
        # gaussian.quats = torch.cat([gaussian.quats, quats], 0)
        # gaussian.opacities = torch.cat([gaussian.opacities, opacities], 0)

        return gaussian

def add_gaussian(gaussian: GaussianModel, num_points: int = 10):

    means = 2 * (torch.rand(num_points, 3, device=gaussian.device) - 0.5)
    gaussian.means = torch.cat([gaussian.means, means], 0)

    scales = torch.rand(num_points, 3, device=gaussian.device)
    gaussian.scales = torch.cat([gaussian.scales, scales], 0)

    rgbs = torch.rand(num_points, 3, device=gaussian.device)
    gaussian.rgbs = torch.cat([gaussian.rgbs, rgbs], 0)

    u = torch.rand(num_points, 1, device=gaussian.device)
    v = torch.rand(num_points, 1, device=gaussian.device)
    w = torch.rand(num_points, 1, device=gaussian.device)
    quats = torch.cat([
        torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
        torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
        torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
        torch.sqrt(u) * torch.cos(2.0 * math.pi * w)
        ], -1)
    gaussian.quats = torch.cat([gaussian.quats, quats], 0)

    opacities = torch.ones((num_points, 1), device=gaussian.device)
    gaussian.opacities = torch.cat([gaussian.opacities, opacities], 0)

    gaussian.means.requires_grad = True
    gaussian.scales.requires_grad = True
    gaussian.rgbs.requires_grad = True
    gaussian.quats.requires_grad = True
    gaussian.opacities.requires_grad = True

    return gaussian