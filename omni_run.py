import torch
from PIL import Image
from pathlib import Path
import os

# 载入模型
from thirdparty.OmniConsistency.src_inference.pipeline import FluxPipeline
from thirdparty.OmniConsistency.src_inference.lora_helper import set_single_lora

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 加载 FLUX.1-dev 基础模型
pipe = FluxPipeline.from_pretrained(
    "/cache/hanmo/models/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
).to(device)

# 2. 加载 OmniConsistency 基础 LoRA（作为一致性插件）
oc_path = "/cache/hanmo/models/omniconsistency/models/OmniConsistency.safetensors"
set_single_lora(
    pipe.transformer,
    oc_path,
    lora_weights=[1.0],
    cond_size=512,
)

# 3. 加载具体风格 LoRA（3D Chibi 只是一个例子）
pipe.unload_lora_weights()  # 先清掉之前可能加载的
lora_dir = "/cache/hanmo/models/omniconsistency/LoRAs/LoRAs"
pipe.load_lora_weights(
    lora_dir,
    weight_name="3D_Chibi_rank128_bf16.safetensors",
)

def stylize_image(
    content_path: str,
    output_path: str = "output.png",
    prompt: str = "3D Chibi style, stylize this image while preserving its content.",
    height: int | None = None,
    width: int | None = None,
    guidance_scale: float = 3.5,
    num_inference_steps: int = 25,
    seed: int = 123,
):
    # 读取内容图像（上传图）
    content_img = Image.open(content_path).convert("RGB")

    # 高宽：默认跟原图接近，但要是 8 的倍数
    if height is None:
        height = content_img.height
    if width is None:
        width = content_img.width

    height = (height // 8) * 8
    width = (width // 8) * 8

    generator = torch.Generator("cpu").manual_seed(seed)

    # 官方空间的用法：spatial_images = [上传的图]，subject_images = [] :contentReference[oaicite:2]{index=2}
    out = pipe(
        prompt=prompt,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        max_sequence_length=512,      # 非常关键！
        generator=generator,
        spatial_images=[content_img], # 只传一张图
        subject_images=[],            # 不要乱加图片，否则容易炸 RoPE
        cond_size=512,
    ).images[0]

    out.save(output_path)
    print(f"Saved stylized image to {output_path}")

if __name__ == "__main__":
    from process import dataloader
    output_dir = "/cache/hanmo/style_output/OmniConsistency"
    from tqdm import tqdm
    for cnt_path, _, prompt, _ in tqdm(dataloader(), total=4225):
        basename = os.path.basename(cnt_path)
        stylize_image(cnt_path, output_path=os.path.join(output_dir, basename))
        
