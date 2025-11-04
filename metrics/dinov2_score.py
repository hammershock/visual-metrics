# metrics/dinov2_score.py
import os
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
from PIL import Image


class DINOv2Score:
    def __init__(self, model_name="facebook/dinov2-large", cache_dir="/cache/hanmo/models/facebook/dinov2-large", device=None):
        """
        Initialize DINOv2 feature extractor using HuggingFace transformers.
        The model will be automatically downloaded (if needed) to /cache/hanmo/models.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Force HuggingFace to use the given cache directory
        os.environ["TRANSFORMERS_CACHE"] = cache_dir

        # Load processor and model
        self.processor = AutoImageProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def _get_embedding(self, image: Image.Image):
        """
        Get normalized embedding from an RGB image.
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        features = outputs.last_hidden_state.mean(dim=1)  # global average pooling
        features = F.normalize(features, dim=-1)
        return features

    def score(self, image1: Image.Image, image2: Image.Image) -> float:
        """
        Compute cosine similarity between two images.
        """
        f1 = self._get_embedding(image1)
        f2 = self._get_embedding(image2)
        similarity = torch.sum(f1 * f2).item()
        return similarity


if __name__ == "__main__":
    # --- Example usage ---
    from PIL import Image
    import requests

    # 示例图片
    url1 = "https://pytorch.org/assets/images/deeplab1.png"
    url2 = "https://pytorch.org/assets/images/deeplab2.png"
    img1 = Image.open(requests.get(url1, stream=True).raw).convert("RGB")
    img2 = Image.open(requests.get(url2, stream=True).raw).convert("RGB")

    scorer = DINOv2Score("/cache/hanmo/models/facebook/dinov2-base")
    sim = scorer.score(img1, img2)
    print(f"Cosine similarity: {sim:.4f}")
