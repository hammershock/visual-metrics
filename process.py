import os
import json

from PIL import Image

def dataloader(base_dir="/temp/rey/stylagent/images/omniconsistency"):
    for category in os.listdir(base_dir):
        category_path = os.path.join(base_dir, category)
        print(category_path)
        # load jsonl:
        analysis_path = os.path.join(category_path, "analysis.jsonl")  # analysis.jsonl | val.jsonl
        with open(analysis_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                assert len(line) > 0
                data = json.loads(line)
        
                cnt_path = base_dir + data['cnt']
                ref_path = base_dir + data['ref']
                ours_sty_path = base_dir + data['sty']
                prompt = data["prompt"]
                # yield Image.open(cnt_path), Image.open(ref_path)
                yield cnt_path, ref_path, prompt, ours_sty_path
        

if __name__ == "__main__":
    for cnt, ref, prompt, _ in dataloader():
        pass
        