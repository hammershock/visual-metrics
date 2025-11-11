import os
import json
import argparse

from PIL import Image
from tqdm import tqdm

from utils import load_jsonl
from metrics import DINOv2Score, VGGScore, FID1KScore


def test_pixel(metrics, cnt_image, sty_image, output_image):
    scores = {
        "DINOv2": metrics["DINOv2"].score(cnt_image, output_image), 
        "StyleLoss": metrics["StyleLoss"].score(sty_image, output_image), 
        "FID": metrics["FID"].score(sty_image, output_image)
    }
    return scores
    

def log_jsonl(data, log_file_path):
    # print(data)
    log_dir = os.path.dirname(log_file_path)
    os.makedirs(log_dir, exist_ok=True)
    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
        f.flush()
                
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="text pixel")
    parser.add_argument("--inp", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()


    output_log_file_path =  args.out # 输出测试指标jsonl
    input_log_data_path = args.inp  # 输入(cnt, sty, output)图像三元组的jsonl（推理过程日志）
    
    metrics = {
        "DINOv2": DINOv2Score(), 
        "StyleLoss": VGGScore(),  
        "FID": FID1KScore()
    }
    
    if os.path.exists(output_log_file_path):
        finished_pairs = {(l["content"], l["style"]) for l in load_jsonl(output_log_file_path)}
    else:
        finished_pairs = set()
        
    log_data = load_jsonl(input_log_data_path)
    
    p_bar = tqdm(log_data)
    for data in p_bar:
        if (data["content"], data["style"]) in finished_pairs:
            continue
        cnt_image = Image.open(data["content"]).convert("RGB")
        sty_image = Image.open(data["style"]).convert("RGB")
        output_image = Image.open(data["output"]).convert("RGB")
        p_bar.set_postfix(size=output_image.size)
        ow, oh = output_image.size; assert ow == oh
        cnt_image = cnt_image.resize((ow, oh), Image.Resampling.LANCZOS)
        sty_image = sty_image.resize((ow, oh), Image.Resampling.LANCZOS)
        
        log_entry = {"content": data["content"], "style": data["style"], "output": data["output"]}
        scores = test_pixel(metrics, cnt_image=cnt_image, sty_image=sty_image, output_image=output_image)
        log_entry.update(scores)
        log_jsonl(log_entry, log_file_path=output_log_file_path)
        finished_pairs.add((data["content"], data["style"]))
