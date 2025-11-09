import os
import random
from itertools import product

if __name__ == "__main__":
    src_dir = "/temp/rey/stylagent/images/omniconsistency" # "./omniconsistency"
    cnt_dir = "/temp/hanmo/data/test2017"  # COCO2017test共40670图片
    categories = os.listdir(src_dir)
    cnt_files = os.listdir(cnt_dir)
    # cnt_files[0]: 000000021655.jpg
    assert len(categories) == 22
    
    datalines = []
    
    for category in categories:
        category_val_tar_dir = os.path.join(src_dir, category, "val", "tar")
        ref_files = os.listdir(category_val_tar_dir)
        # resample 100 ref images
        ref_files_resample = random.choices(ref_files, k=100)
        ref_paths = [os.path.join(category_val_tar_dir, ref_file) for ref_file in ref_files_resample]
        # ref_paths[0]:  /temp/rey/stylagent/images/omniconsistency/3D_Chibi/val/tar/110.png
        
        # resample 100 cnt images from COCO2017test
        cnt_files_resample = random.choices(cnt_files, k=100)
        cnt_paths = [os.path.join(cnt_dir, cnt_file) for cnt_file in cnt_files_resample]
        
        # make 100*100 pairs
        for ref_path, cnt_path in product(ref_paths, cnt_paths):
            datalines.append({
                "cnt_path": cnt_path, 
                "ref_path": ref_path, 
                "category": category
            })
            
    print(len(datalines))  # 220000