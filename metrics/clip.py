import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


class CLIPScore:
    def __init__(self, model_name="openai/clip-vit-large-patch14", cache_dir="/cache/hanmo/models", device=None):
        """
        用于计算两张图像之间的 CLIP Score。
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = CLIPModel.from_pretrained(model_name, cache_dir=cache_dir).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        self.model.eval()

    @torch.no_grad()
    def _get_image_embedding(self, image: Image.Image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        img_feat = self.model.get_image_features(**inputs)
        img_feat = F.normalize(img_feat, dim=-1)
        return img_feat

    def score(self, image1: Image.Image, image2: Image.Image):
        f1 = self._get_image_embedding(image1)
        f2 = self._get_image_embedding(image2)
        return torch.sum(f1 * f2).item()
