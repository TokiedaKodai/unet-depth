import numpy as np
# import cv2
from tqdm import tqdm
import random

from keras.utils import Sequence

import info as I
# import depth_tools as DT


# is_aug_lumi = True
# is_aug_lumi = False
lumi_scale_range = [0.5, 1.5]
class MiniBatchGenerator(Sequence):
    def __init__(self, dir_name, data_num, use_num, is_aug_lumi=False):
        self.data_size = data_num
        self.batches_per_epoch = use_num
        self.x_file = dir_name + '/x/{:05d}.npy'
        self.y_file = dir_name + '/y/{:05d}.npy'
        self.is_aug_lumi = is_aug_lumi

    def __getitem__(self, idx):
        random_idx = random.randrange(0, self.data_size)
        x_batch = np.load(self.x_file.format(random_idx))
        y_batch = np.load(self.y_file.format(random_idx))
        if self.is_aug_lumi:
            aug_scale = random.uniform(lumi_scale_range[0], lumi_scale_range[1])
            x_batch[:, :, :, 0] *= aug_scale
            x_batch[:, :, :, 2] *= aug_scale
        return x_batch, y_batch

    def __len__(self):
        return self.batches_per_epoch

    def on_epoch_end(self):
        pass

# def LoadData(dir_data, range_data, savefile):
#     def clip_patch(img, top_left, size):
#         t, l, h, w = *top_left, *size
#         return img[t:t + h, l:l + w]
    
#     path_gt = savefile['gt']
#     path_depth = savefile['depth']
#     path_shade = savefile['shade']
#     path_proj = savefile['proj']

#     list_x = []
#     list_y = []

#     for idx in tqdm(range_data):
#         file_gt = path_gt.format(idx)
#         file_depth = path_depth.format(idx)
#         file_shade = path_shade.format(idx)
#         file_proj = path_proj.format(idx)
#         # Read Image
#         img_gt = cv2.imread(file_gt, -1)
#         img_depth = cv2.imread(file_depth, -1)
#         img_shade = cv2.imread(file_shade, 0) # GrayScale
#         img_proj = cv2.imread(file_proj, 0) # GrayScale
#         # Clip Image
#         img_gt = img_gt[:I.shape_img[0], :I.shape_img[1], :]
#         img_depth = img_depth[:I.shape_img[0], :I.shape_img[1], :]
#         img_shade = img_shade[:I.shape_img[0], :I.shape_img[1]]
#         img_proj = img_proj[:I.shape_img[0], :I.shape_img[1]]
#         # GT
#         gt = DT.unpack_bmp_bgra_to_float(img_gt)
#         # Depth
#         depth = DT.unpack_bmp_bgra_to_float(img_depth)

