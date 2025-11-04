import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from PIL import Image
from torchvision import transforms



IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def normalize_vgg(x: torch.Tensor) -> torch.Tensor:
    """Normalize input image to VGG / ImageNet statistics.
    Expects x in range [0,1]. Returns normalized tensor."""
    return (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)


def gram_matrix(features: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """
    Compute Gram matrix for a batch of feature maps.
    features: (B, C, H, W)
    returns: (B, C, C)
    If normalize=True, divide by (C * H * W) to make loss scale-invariant.
    """
    B, C, H, W = features.shape
    F = features.view(B, C, H * W)  # (B, C, N)
    G = torch.bmm(F, F.transpose(1, 2))  # (B, C, C)
    if normalize:
        return G / (C * H * W)
    return G


class VGG16FeatureExtractor(nn.Module):
    """Extract intermediate conv activations from torchvision.models.vgg16.features.
    Default layers: conv1_1, conv2_1, conv3_1, conv4_1, conv5_1.
    """

    DEFAULT_LAYERS = {
        "conv1_1": 1,
        "conv1_2": 3,
        "conv2_1": 6,
        "conv2_2": 8,
        "conv3_1": 11,
        "conv3_2": 13,
        "conv3_3": 15,
        "conv4_1": 18,
        "conv4_2": 20,
        "conv4_3": 22,
        "conv5_1": 25,
        "conv5_2": 27,
        "conv5_3": 29,
    }

    def __init__(self, layers=None):
        super().__init__()
        self.vgg = models.vgg16(weights=torchvision.models.vgg.VGG16_Weights.DEFAULT).features.eval()
        for p in self.vgg.parameters():
            p.requires_grad = False

        if layers is None:
            self.layers = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]
        else:
            self.layers = layers
        self.layer_indices = [self.DEFAULT_LAYERS[name] for name in self.layers]

    def forward(self, x: torch.Tensor):
        """Return dict of activations keyed by layer name."""
        activations = {}
        out = x
        max_idx = max(self.layer_indices)
        for i, layer in enumerate(self.vgg):
            out = layer(out)
            if i in self.layer_indices:
                name = next(k for k, v in self.DEFAULT_LAYERS.items() if v == i)
                activations[name] = out
            if i >= max_idx:
                break
        return activations


class PerceptualLoss(nn.Module):
    """Compute style loss (Gram-MSE) between input and a style image.
    style_image should be in range [0,1] shape (C,H,W) or (B,C,H,W).
    """

    def __init__(self, layers=None, layer_weights=None, normalize_gram=True):
        super().__init__()
        self.extractor = VGG16FeatureExtractor(layers=layers)
        self.normalize_gram = normalize_gram

        if layer_weights is None:
            self.layer_weights = {name: 1.0 for name in self.extractor.layers}
        else:
            self.layer_weights = layer_weights

        # self.loss_fn = nn.MSELoss(reduction="mean")
        self.loss_fn = nn.L1Loss(reduction="mean")

    def forward(self, predict_images: torch.Tensor, target_images: torch.Tensor):
        predict_norm = normalize_vgg(predict_images)
        target_norm = normalize_vgg(target_images)
        predict_feats = self.extractor(predict_norm)
        target_feats = self.extractor(target_norm)
        loss = 0.0
        for name in predict_feats.keys():
            predict_feat, target_feat = predict_feats[name], target_feats[name]
            predict_G = gram_matrix(predict_feat, normalize=self.normalize_gram)
            target_G = gram_matrix(target_feat, normalize=self.normalize_gram)
            weight = self.layer_weights.get(name, 1.0)
            loss_part = self.loss_fn(predict_G, target_G)
            loss = loss + weight * loss_part
        return loss


class VGGScore:
    def __init__(self, model_name="VGG16", layers=None, layer_weights=None, normalize_gram=None):
        # unused: model_name
        self.loss = PerceptualLoss(layers=layers, layer_weights=layer_weights, normalize_gram=normalize_gram)
        self.to_tensor = transforms.ToTensor()
        
    def score(self, image1: Image, image2: Image) -> float:
        return self.loss(self.to_tensor(image1), self.to_tensor(image2)).item()