import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image


class FID1KScore:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # 加载 InceptionV3（ImageNet 预训练）
        inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        inception.fc = torch.nn.Identity()  # 移除分类头
        inception.eval().to(self.device)
        self.model = inception

        # 与 FID 一致的预处理
        self.preprocess = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def _get_feature(self, image: Image.Image):
        x = self.preprocess(image).unsqueeze(0).to(self.device)
        feat = self.model(x)
        feat = F.normalize(feat, dim=-1)
        return feat

    def score(self, img1: Image.Image, img2: Image.Image):
        f1 = self._get_feature(img1)
        f2 = self._get_feature(img2)
        sim = torch.sum(f1 * f2).item()
        return sim
