import torch
from PIL import Image
from pathlib import Path
# 载入模型
from thirdparty.OmniConsistency.src_inference.pipeline import FluxPipeline
from thirdparty.OmniConsistency.src_inference.lora_helper import set_single_lora


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load base model
pipe = FluxPipeline.from_pretrained("/cache/hanmo/models/FLUX.1-dev", torch_dtype=torch.bfloat16).to(device)

# 加载 OmniConsistency 模型 LoRA 权重
set_single_lora(pipe.transformer, "/cache/hanmo/models/omniconsistency/models/OmniConsistency.safetensors", lora_weights=[1.0], cond_size=512)

pipe.unload_lora_weights()
lora_path = "/cache/hanmo/models/omniconsistency/LoRAs/LoRAs"
pipe.load_lora_weights(lora_path,  weight_name="3D_Chibi_rank128_bf16.safetensors")

if __name__ == "__main__":
    content_path = "./test_images/cnt.png"  # 内容图像
    style_path   = "./test_images/ref.png"  # 风格图像
    content_img = Image.open(content_path).convert("RGB")
    style_img   = Image.open(style_path).convert("RGB")

    # 你也许希望将风格图像作为 prompt 参考或使用 spatial_images 参数
    prompt = "Transfer the semantic style from Picture 2 to Picture 1:"

    # 推理
    out = pipe(
        prompt=prompt,
        height=content_img.height,
        width =content_img.width,
        guidance_scale=3.5,
        num_inference_steps=25,
        generator=torch.Generator(device="cpu").manual_seed(123),
        spatial_images=[style_img],
        subject_images=[content_img],
        cond_size=512,
    ).images[0]

    # 保存结果
    out.save("output.png")
    print("Saved stylized image to output.png")
