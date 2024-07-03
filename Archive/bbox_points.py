from tqdm import tqdm
import os
import numpy as np
import torch
from torchvision.io import read_image
from torchvision.ops import masks_to_boxes

# ASSETS_DIRECTORY = "C:/dataset/TOD/training_set"
# dir = os.listdir(ASSETS_DIRECTORY)
# with tqdm(total=len(dir)*6) as pbar:
#     for folder in dir:
#         folder_path = os.path.join(ASSETS_DIRECTORY, folder)
#         for i in range(1,7):
#             img_path = os.path.join(folder_path, f"rgb_0000{i}.jpeg")
#             mask_path = os.path.join(folder_path, f"segmentation_0000{i}.png")
#             img = read_image(img_path)
#             mask = read_image(mask_path)

#             # We get the unique colors, as these would be the object ids.
#             obj_ids = torch.unique(mask)

#             # first id is the background, so remove it.
#             obj_ids = obj_ids[1:]

#             # split the color-encoded mask into a set of boolean masks.
#             # Note that this snippet would work as well if the masks were float values instead of ints.
#             masks = mask == obj_ids[:, None, None]

#             boxes = masks_to_boxes(masks).detach().numpy()
#             fname=f"{folder_path}/bbox_0000{i}.txt"
#             np.savetxt(fname, boxes)            
#             pbar.update(1)

ASSETS_DIRECTORY = "C:/dataset/TOD/training_set"
dir = os.listdir(ASSETS_DIRECTORY)
for folder in tqdm(dir):
    folder_path = os.path.join(ASSETS_DIRECTORY, folder)
    
    img_path = os.path.join(folder_path, f"rgb_00000.jpeg")
    mask_path = os.path.join(folder_path, f"segmentation_00000.png")
    img = read_image(img_path)
    mask = read_image(mask_path)

    # We get the unique colors, as these would be the object ids.
    obj_ids = torch.unique(mask)

    # first id is the background, so remove it.
    obj_ids = obj_ids[1:]

    # split the color-encoded mask into a set of boolean masks.
    # Note that this snippet would work as well if the masks were float values instead of ints.
    masks = mask == obj_ids[:, None, None]

    boxes = masks_to_boxes(masks).detach().numpy()
    fname=f"{folder_path}/bbox_00000.txt"
    np.savetxt(fname, boxes)            

