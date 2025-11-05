import torch
from PIL import Image
from pathlib import Path
import os
import json

from tqdm import tqdm

# 载入模型
from thirdparty.OmniConsistency.src_inference.pipeline import FluxPipeline
from thirdparty.OmniConsistency.src_inference.lora_helper import set_single_lora


def stylize_image(
    content_path: str,
    output_path: str = "output.png",
    prompt: str = "3D Chibi style, stylize this image while preserving its content.",
    height: int | None = None,
    width: int | None = None,
    guidance_scale: float = 3.5,
    num_inference_steps: int = 25,
    seed: int = 42,
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


style_to_lora = {
    "3D_Chibi":         "3D_Chibi_rank128_bf16.safetensors",
    "American_Cartoon": "American_Cartoon_rank128_bf16.safetensors",
    "Chinese_Ink":      "Chinese_Ink_rank128_bf16.safetensors",
    "Clay_Toy":         "Clay_Toy_rank128_bf16.safetensors",
    "Fabric":           "Fabric_rank128_bf16.safetensors",
    "Ghibli":           "Ghibli_rank128_bf16.safetensors",
    "Irasutoya":        "Irasutoya_rank128_bf16.safetensors",
    "Jojo":             "Jojo_rank128_bf16.safetensors",
    "LEGO":             "LEGO_rank128_bf16.safetensors",
    "Line":             "Line_rank128_bf16.safetensors",
    "Macaron":          "Macaron_rank128_bf16.safetensors",
    "Oil_Painting":     "Oil_Painting_rank128_bf16.safetensors",
    "Origami":          "Origami_rank128_bf16.safetensors",
    "Paper_Cutting":    "Paper_Cutting_rank128_bf16.safetensors",
    "Picasso":          "Picasso_rank128_bf16.safetensors",
    "Pixel":            "Pixel_rank128_bf16.safetensors",
    "Poly":             "Poly_rank128_bf16.safetensors",
    "Pop_Art":          "Pop_Art_rank128_bf16.safetensors",
    "Rick_Morty":       "Rick_Morty_rank128_bf16.safetensors",
    "Snoopy":           "Snoopy_rank128_bf16.safetensors",
    "Van_Gogh":         "Van_Gogh_rank128_bf16.safetensors",
    "Vector":           "Vector_rank128_bf16.safetensors",
}



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flux_dir = "/cache/hanmo/models/FLUX.1-dev"
    oc_path = "/cache/hanmo/models/omniconsistency/models/OmniConsistency.safetensors"
    lora_dir = "/cache/hanmo/models/omniconsistency/LoRAs/LoRAs"
    output_dir = "/cache/hanmo/style_output/OmniConsistency/full_prompt"
    base_dir="/temp/rey/stylagent/images/omniconsistency"
    
    # 1. 加载 FLUX.1-dev 基础模型
    pipe = FluxPipeline.from_pretrained(flux_dir, torch_dtype=torch.bfloat16,).to(device)

    # 2. 加载 OmniConsistency 基础 LoRA（作为一致性插件）
    set_single_lora(
        pipe.transformer,
        oc_path,
        lora_weights=[1.0],
        cond_size=512,
    )

    categories = os.listdir(base_dir)
    p_bar = tqdm(categories, total=len(categories))
    for category in p_bar:
        category_path = os.path.join(base_dir, category)
        p_bar.set_description(f"Processing {category}")
        
        # load lora
        pipe.unload_lora_weights()  # 先清掉之前可能加载的
        pipe.load_lora_weights(
            lora_dir,
            weight_name=style_to_lora[category],  # "3D_Chibi_rank128_bf16.safetensors"
        )
        os.makedirs(os.path.join(output_dir, category), exist_ok=True)
        # load jsonl:
        analysis_path = os.path.join(category_path, "analysis.jsonl")  # analysis.jsonl | val.jsonl
        with open(analysis_path, "r", encoding="utf-8") as f:
            jsons = []
            for line in f:
                line = line.strip()
                jsons.append(json.loads(line))
                assert len(line) > 0
            
            for i, data in enumerate(jsons):
                p_bar.set_postfix(progress=f"({i+1}/{len(jsons)})")
                cnt_path = base_dir + data['cnt']
                ref_path = base_dir + data['ref']
                ours_sty_path = base_dir + data['sty']
                
                base_prompt = "Transfer the semantic style of Picture 1"
                description = data["prompt"].split('\n')[1]
                # prompt = base_prompt + " into following description" + description  # use full prompt
                prompt = base_prompt  # use short prompt
                
 
                # generate
                basename = os.path.basename(cnt_path)
                basename = os.path.basename(cnt_path)
                output_path = os.path.join(output_dir, category, basename)
                
                if not os.path.exists(output_path):
                    stylize_image(cnt_path, output_path=output_path, width=512, height=512, prompt=prompt)