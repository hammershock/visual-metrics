from metrics import FID1KScore, CLIPScore
import os
import json
from PIL import Image

# conda activate hanmo-metrics

if __name__ == "__main__":
    fid1k = FID1KScore()
    clip = CLIPScore()

    base_dir = "/temp/hanmo/style_output/OmniConsistency/full_prompt"
    gt_dir = "/temp/hanmo/style_output/GT"
    log_path = "./logs/semantic_omniconsistency_full_prompt_log.jsonl"
    cnt = 0
    with open(log_path, "w", encoding="utf-8") as f_log:
        for root, dirs, files in os.walk(base_dir):
            for category in dirs:
                image_dir = os.path.join(base_dir, category)
                for image_file in os.listdir(image_dir):
                    image_path = os.path.join(image_dir, image_file)
                    gt_path = os.path.join(gt_dir, category, image_file)
                    cnt += 1
                    # assert os.path.exists(image_path)
                    # assert os.path.exists(gt_path)
                    
                    if not (os.path.exists(image_path) and os.path.exists(gt_path)):
                        print(f"⚠️ 跳过不存在的文件：{image_path} 或 {gt_path}")
                        continue

                    test_image = Image.open(image_path).convert("RGB")
                    gt_image = Image.open(gt_path).convert("RGB")

                    fid_score = fid1k.score(test_image, gt_image)
                    clip_score = clip.score(test_image, gt_image)

                    log_entry = {
                        "image_path": image_path,
                        "gt_path": gt_path,
                        "FID1K": float(fid_score),
                        "CLIP": float(clip_score),
                    }

                    f_log.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                    f_log.flush()
                    os.fsync(f_log.fileno())  # 强制刷新到磁盘

                    print(f"({cnt}/845)[{category}] {image_file} → FID1K={fid_score:.4f}, CLIP={clip_score:.4f}")

    print(f"\n✅ 日志已保存到 {log_path}")
