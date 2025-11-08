import os
import json
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from metrics import DINOv2Score, VGGScore, FID1KScore


if __name__ == "__main__":
    cnt_dir = "/temp/hanmo/data/pairs2k/COCO_subset"
    sty_dir = "/temp/hanmo/data/pairs2k/Style30k_subset"
    target_dir = "/temp/hanmo/style_output/StyleSSP/pairs2k"
    log_file_path = "./logs/pixel_stylessp_log.jsonl"

    # ========== 1️⃣ 加载已完成记录 ==========
    finished_pairs = set()
    if os.path.exists(log_file_path):
        with open(log_file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    key = (Path(record["content"]).name, Path(record["style"]).name)
                    finished_pairs.add(key)
                except Exception:
                    continue
        print(f"✅ 已检测到 {len(finished_pairs)} 条历史记录，将跳过这些组合。")

    cnt_files = os.listdir(cnt_dir)
    sty_files = os.listdir(sty_dir)
    assert len(os.listdir(target_dir)) == 2000  # 2k image pairs
    ext = ".png"
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # ========== 2️⃣ 初始化度量 ==========
    dino = DINOv2Score()
    vgg = VGGScore()
    fid = FID1KScore()

    # ========== 3️⃣ 主循环 ==========
    for i, cnt_file in enumerate(cnt_files):
        for sty_file in tqdm(sty_files, desc=f"Processing ({i+1}/{len(cnt_files)})"):
            key = (cnt_file, sty_file)
            if key in finished_pairs:
                continue  # ✅ 跳过已处理组合

            cnt_path = os.path.join(cnt_dir, cnt_file)
            sty_path = os.path.join(sty_dir, sty_file)
            target_file = f"{Path(cnt_file).stem}@{Path(sty_file).stem}" + ext
            target_path = os.path.join(target_dir, target_file)
            if not os.path.isfile(target_path):
                print(f"⚠️ {target_path} not found, skip.")
                continue

            # ========== 计算各指标 ==========
            cnt_image = Image.open(cnt_path).convert("RGB")
            sty_image = Image.open(sty_path).convert("RGB")
            target_image = Image.open(target_path).convert("RGB")

            dinov2_score = dino.score(cnt_image, target_image)
            vgg_score = vgg.score(sty_image, target_image)
            fid_score = fid.score(sty_image, target_image)

            data = {
                "content": cnt_path,
                "style": sty_path,
                "output": target_path,
                "DINOv2": dinov2_score,
                "StyleLoss": vgg_score,
                "FID": fid_score
            }

            # ========== 写入日志 ==========
            with open(log_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
                f.flush()

            finished_pairs.add(key)  # 即时记录，防止中途重复
