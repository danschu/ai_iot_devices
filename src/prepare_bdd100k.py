import json
import os
import cv2
from PIL import Image, ImageDraw, ImagePalette
from seaborn import color_palette as create_color_palette
import numpy as np
import glob
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor


    
"""
  Create the segmentation masks from the bbox with SAM2
  (We could use the segmentation masks from bdd100k, but then we only have 10k instead of 100k images)  

"""

predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")

class_list = ["person", "car", "bus", "truck", "rider", "train"]
max_files_dataset = {"train": 1000, "val": 100}


cats = []
color_palette = []
for color in create_color_palette(n_colors=256):
    color_palette.append(int(color[0]*255))
    color_palette.append(int(color[1]*255))
    color_palette.append(int(color[2]*255))
color_palette = ImagePalette.ImagePalette(mode='RGB', palette=color_palette)
    
# reduce number of images for testing
os.makedirs("./dataset/masks", exist_ok=True)

for dataset_typ in ["train", "val"]:
    print(f"Preparing data for '{dataset_typ}'")
    with open(f"./dataset/{dataset_typ}_list.txt", "w") as outlist:
        
        max_files = max_files_dataset[dataset_typ]
        cnt = 0

        for image_file in glob.glob(f"./dataset/bdd100k/images/100k/{dataset_typ}/*.jpg"):
            cnt += 1
            if cnt > max_files:
                break
            print(f"Processing image file {image_file} [{cnt}/{max_files}]")
            
            basefile = os.path.basename(image_file)
            basefile, _ = os.path.splitext(basefile) 
            image = cv2.imread(image_file)
            h, w, _c = image.shape
            mask = np.zeros((h, w), np.uint8)
            mask_file_full = os.path.abspath(f"./dataset/masks/{basefile}.png")
            image_file_full = os.path.abspath(image_file)
            
            outlist.write(f"{image_file_full} {mask_file_full}\n")
            if os.path.exists(mask_file_full):
                continue
            predictor.set_image(image)
            with open(f"./dataset/100k/{dataset_typ}/{basefile}.json") as f:
                data = json.loads(f.read())
                for obj in data["frames"][0]["objects"]:
                    cat = obj["category"]
                    if cat not in cats:
                        print(f"New category: [{cat}]")
                        cats.append(cat)
                    if cat in class_list:
                        x1, y1, x2, y2 = obj["box2d"]["x1"],obj["box2d"]["y1"],obj["box2d"]["x2"],obj["box2d"]["y2"]

                        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                            masks, _, _ = predictor.predict(box=[x1,y1,x2,y2])
                            
                            c, h, w = masks.shape
                            for idx in range(c):
                                
                                one_mask = masks[idx,:,:]
                                mask[one_mask > 0] = 1
                                

            mask = Image.fromarray(mask)
            mask_pil = Image.new("P", (w, h))
            mask_pil.putpalette(color_palette)
            mask_pil.paste(mask, (0, 0))
            mask_pil.save(mask_file_full)