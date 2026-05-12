"""
head_init.py — Final layer initialization (student-implemented).

Students: Implement `init_last_layer` to control how the new classification
head is initialized before fine-tuning begins. The skeleton below uses
Kaiming uniform weights and zero bias — you are expected to experiment with
alternatives (e.g. Xavier, orthogonal, small-scale random, learned bias init).
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as T
from torch.utils.data import DataLoader


_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR100_STD  = (0.2675, 0.2565, 0.2761)
_DATA_DIR = "./data"
_RIDGE_LAMBDA = 1e-2


def init_last_layer(layer: nn.Linear) -> None:
    """Initialize the weights and bias of the final classification layer in-place.

    This function is called once during model construction (see model.py).
    Modify it to experiment with different initialization strategies and observe
    their effect on the "initialized head" evaluation checkpoint.

    Args:
        layer: The ``nn.Linear`` layer that serves as the new CIFAR100 head.
               Modifies the layer in-place; return value is ignored.

    Student task:
        Replace or extend the skeleton below. Some strategies to consider:
          - ``nn.init.xavier_uniform_``  — preserves variance across layers
          - ``nn.init.orthogonal_``      — encourages diverse feature directions
          - Small-scale init (e.g. scale weights by 0.01) — conservative start
          - Non-zero bias init           — useful when class priors are known
    """
    # -------------------------------------------------------------------------
    # STUDENT: Replace or extend the initialization below.
    # -------------------------------------------------------------------------
    _ridge_init(layer)
    # -------------------------------------------------------------------------


def _ridge_init(layer: nn.Linear) -> None:
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    backbone.fc = nn.Identity()
    backbone.eval().to(device)

    transform = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
    ])

    dataset = datasets.CIFAR100(
        root=_DATA_DIR, train=True, download=True, transform=transform
    )
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)

    features, labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            features.append(backbone(imgs.to(device)).cpu())
            labels.append(lbls)

    X = torch.cat(features, dim=0).float()
    y = torch.cat(labels,   dim=0).long()

    N, D = X.shape
    C = 100

    Y = torch.zeros(N, C)
    Y.scatter_(1, y.unsqueeze(1), 1.0)

    A = X.T @ X + _RIDGE_LAMBDA * N * torch.eye(D)
    B = X.T @ Y
    W = torch.linalg.solve(A, B)

    residuals = Y - X @ W
    bias = residuals.mean(dim=0)

    with torch.no_grad():
        layer.weight.copy_(W.T)
        layer.bias.copy_(bias)
