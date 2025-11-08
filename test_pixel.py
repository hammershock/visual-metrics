import os
import json
from PIL import Image
from pathlib import Path

from tqdm import tqdm

from metrics import DINOv2Score, VGGScore, FID1KScore


if __name__ == "__main__":
    cnt_dir = "/temp/hanmo/data/pairs2k/COCO_subset"
    sty_dir = "/temp/hanmo/data/pairs2k/Style30k_subset"
    cnt_files = os.listdir(cnt_dir)
    sty_files = os.listdir(sty_dir)
    # print(len(cnt_files), len(sty_files))  # 80, 25
    target_dir = "/temp/hanmo/style_output/StyleSSP/pairs2k"
    assert len(os.listdir(target_dir)) == 2000  # 2k image pairs
    ext = ".png"
    
    log_file_path = "./logs/pixel_stylessp_log.jsonl"
    
    dino = DINOv2Score()
    vgg = VGGScore()
    fid = FID1KScore()
    
    
    for i, cnt_file in enumerate(cnt_files):
        for sty_file in tqdm(sty_files, desc=f"Processing ({i+1}/{len(cnt_files)})"):
            cnt_path = os.path.join(cnt_dir, cnt_file)
            sty_path = os.path.join(sty_dir, sty_file)
            cnt_image = Image.open(cnt_path).convert("RGB")
            sty_image = Image.open(sty_path).convert("RGB")
            target_file = f"{Path(cnt_file).stem}@{Path(sty_file).stem}" + ext
            target_path = os.path.join(target_dir, target_file)
            assert os.path.isfile(target_path), f"{target_path} does not exists!"
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
            
            with open(log_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
                f.flush()
            
            
            
            
            
    