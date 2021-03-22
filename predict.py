import cv2
import numpy as np
from itertools import product
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.cm import ScalarMappable
import io
import sys
import pandas as pd
import argparse

import network
import depth_tools
import common_tools
import compare_error
import config as cf

'''
ARGV
1: output dir
2: epoch num
3: data type
'''
# Parser
parser = argparse.ArgumentParser()
parser.add_argument('name', help='model name to use training and test')
parser.add_argument('epoch', type=int, help='end epoch num')
parser.add_argument('data', type=int, help='[0 - 3]: data type')
args = parser.parse_args()

out_dir = args.name
epoch_num = args.epoch
data_type = args.data

out_dir = '../output/output_' + out_dir


# normalization
is_shading_norm = False

# parameters
depth_threshold = 0.2
difference_threshold = 0.005
patch_remove = 0.5


# input
is_input_depth = True
is_input_frame = True
is_input_coord = False

img_size = 1200

#select model
save_period = 1

# select_range = save_period * 10
select_range = epoch_num / 2
select_range = 300

is_edge_crop = True
mask_edge_size = 4

patch_rate = 50
patch_rate = 95 # %


if data_type is '0':
    src_dir = '../data/board'
    predict_dir = out_dir + '/predict_{}_board'.format(epoch_num)
    predict_dir = out_dir + '/predict_{}_board-clip'.format(epoch_num)
    data_num = 68
    data_idx_range = list(range(data_num))
elif data_type is '1':
    src_dir = '../data/render'
    src_dir = '../data/render_wave1_300'
    predict_dir = out_dir + '/predict_{}_1wave'.format(epoch_num)
    data_num = 200
    test_num = 20
    data_idx_range = list(range(data_num, data_num + test_num))
    difference_threshold = 0.1
elif data_type is '2':
    src_dir = '../data/render_wave2_1100'
    predict_dir = out_dir + '/predict_{}_2wave'.format(epoch_num)
    data_num = 1000
    test_num = 40
    data_idx_range = list(range(data_num, data_num + test_num))
    difference_threshold = 0.1
elif data_type is '3':
    src_dir = '../data/real'
    predict_dir = out_dir + '/predict_{}_real'.format(epoch_num)
    data_num = 19
    data_idx_range = list(range(data_num))
    mask_edge_size = 2
    patch_rate = 50


# save predict depth PLY file
is_save_ply = True
is_masked_ply = True
is_save_diff = False
is_save_depth_img = True

# predict normalization
is_predict_norm = True # Difference Normalization
is_predict_norm_local = True # Difference Normalization Local
norm_patch_size = 12
is_norm_local_pix = False
norm_patch_size = 13
norm_patch_size = 6
norm_patch_size = 12
norm_patch_size = 24

is_fix_inv_local = False
is_pred_ajust = False

# select from val loss
is_select_val = True

# select minimum loss model
is_select_min_loss_model = True

# Reverse #############################
is_pred_reverse = False
is_pred_pix_reverse = False
is_reverse_threshold = False
is_pred_patch_inverse = False

r_thre = 0.001
#######################################
is_pred_smooth = False

if is_predict_norm:
    predict_dir += '_norm'
    if is_predict_norm_local:
        predict_dir += '-local'
        if is_norm_local_pix:
            predict_dir += '-pix'
        predict_dir += '=' + str(norm_patch_size)
        predict_dir += '_rate=' + str(patch_rate)
    if is_fix_inv_local:
        predict_dir += '_inv-local'
if is_pred_ajust:
    predict_dir += '_ajust'
if is_pred_smooth:
    predict_dir += '_smooth'
if is_pred_reverse:
    predict_dir += '_inverse'
    if is_pred_pix_reverse:
        predict_dir += '-pix'
    elif is_pred_patch_inverse:
        predict_dir += '-patch'
if is_reverse_threshold:
    predict_dir += '_thre=' + str(r_thre)
if is_edge_crop:
    predict_dir += '_crop=' + str(mask_edge_size)

if is_select_val:
    predict_dir += '_vloss'
else:
    predict_dir += '_tloss'

if is_select_min_loss_model:
    predict_dir += '_min'


'''
Test Data
ori 110cm : 16 - 23
small 100cm : 44 - 47
mid 110cm : 56 - 59
'''
# train data
# test data
if data_type is '0':
    train_range = list(range(16))
    data_idx_range = list(range(40, 56))
    test_range = list(range(40, 56))
elif data_type is '1':
    train_range = list(range(data_num))
    test_range = data_idx_range
elif data_type is '2':
    train_range = list(range(data_num))
    test_range = data_idx_range
elif data_type is '3':
    train_range = list()
    test_range = list(range(19))


# save ply range
save_ply_range = test_range
save_img_range = test_range


vmin, vmax = (0.8, 1.4)
vm_range = 0.03
vm_e_range = 0.002

batch_shape = (1200, 1200)
batch_tl = (0, 0)  # top, left

# calib 200317
cam_params = {
    'focal_length': 0.037009,
    'pix_x': 1.25e-05,
    'pix_y': 1.2381443057539635e-05,
    'center_x': 790.902,
    'center_y': 600.635
}
# calib 200427
# cam_params = {
#     'focal_length': 0.036917875,
#     'pix_x': 1.25e-05,
#     'pix_y': 1.2416172558410155e-05,
#     'center_x': 785.81,
#     'center_y': 571.109
# }

def main():
    if data_type is '0':
        src_rec_dir = src_dir + '/rec'
        src_frame_dir = src_dir + '/frame'
        src_gt_dir = src_dir + '/gt'
        src_shading_dir = src_dir + '/shading'
    elif data_type is '1':
        src_frame_dir = src_dir + '/proj'
        src_gt_dir = src_dir + '/gt'
        src_shading_dir = src_dir + '/shade'
        src_rec_dir = src_dir + '/rec'
    elif data_type is '2':
        src_frame_dir = src_dir + '/proj'
        src_gt_dir = src_dir + '/gt'
        src_shading_dir = src_dir + '/shade'
        src_rec_dir = src_dir + '/rec'
    elif data_type is '3':
        src_frame_dir = src_dir + '/proj'
        src_gt_dir = src_dir + '/gt'
        src_shading_dir = src_dir + '/shade'
        src_rec_dir = src_dir + '/rec'

    if is_input_depth:
        if is_input_frame:
            ch_num = 3
            if is_input_coord:
                ch_num = 5
        else:
            ch_num = 2
    else:
        if is_input_frame:
            ch_num = 2
        else:
            ch_num = 1

    # model configuration
    model = network.build_unet_model(batch_shape, ch_num)
    
    # log
    df_log = pd.read_csv(out_dir + '/training.log')
    if is_select_val:
        df = df_log['val_loss']
    else:
        df = df_log['loss']
    
    df.index = df.index + 1
    if is_select_min_loss_model:
        df_select = df[df.index>epoch_num-select_range]
        df_select = df_select[df_select.index<=epoch_num]
        df_select = df_select[df_select.index%save_period==0]
        min_loss = df_select.min()
        idx_min_loss = df_select.idxmin()
        model.load_weights(out_dir + '/model/model-%03d.hdf5'%idx_min_loss)
        # model.load_weights(out_dir + '/model/model-best.hdf5')
    else:
        # model.load_weights(out_dir + '/model-final.hdf5')
        model.load_weights(out_dir + '/model/model-%03d.hdf5'%epoch_num)
    
    # loss graph
    lossgraph_dir = predict_dir + '/loss_graph'
    os.makedirs(lossgraph_dir, exist_ok=True)
    arr_loss = df.values
    arr_epoch = df.index
    if is_select_val:
        init_epochs = [0, 10, int(epoch_num / 2), epoch_num - 200]
    else:
        init_epochs = [0, 10, epoch_num - 200, epoch_num - 100]

    for init_epo in init_epochs:
        if init_epo < 0:
            continue
        if init_epo >= epoch_num:
            continue
        plt.figure()
        plt.plot(arr_epoch[init_epo: epoch_num], arr_loss[init_epo: epoch_num])
        if is_select_min_loss_model:
            plt.plot(idx_min_loss, min_loss, 'ro')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('{} : epoch {} - {}'.format(df.name, init_epo + 1, epoch_num))
        plt.savefig(lossgraph_dir + '/loss_{}-{}.pdf'.format(init_epo + 1, epoch_num))

    # error compare txt
    err_strings = 'index,type,MAE depth,MAE predict,RMSE depth,RMSE predict,SD AE depth,SD AE predict,SD RSE depth,SD RSE predict\n'

    os.makedirs(predict_dir, exist_ok=True)
    for test_idx in tqdm(data_idx_range):
        if data_type is '0':
            src_bgra = src_frame_dir + '/frame{:03d}.png'.format(test_idx)
            src_depth_gap = src_rec_dir + '/depth{:03d}.bmp'.format(test_idx)
            src_depth_gt = src_gt_dir + '/gt{:03d}.bmp'.format(test_idx)
            src_shading = src_shading_dir + '/shading{:03d}.bmp'.format(test_idx)
        elif data_type is '1':
            src_bgra = src_frame_dir + '/{:05d}.png'.format(test_idx)
            src_depth_gt = src_gt_dir + '/{:05d}.bmp'.format(test_idx)
            src_shading = src_shading_dir + '/{:05d}.png'.format(test_idx)
            src_depth_gap = src_rec_dir + '/{:05d}.bmp'.format(test_idx)
        elif data_type is '2':
            src_bgra = src_frame_dir + '/{:05d}.png'.format(test_idx)
            src_depth_gt = src_gt_dir + '/{:05d}.bmp'.format(test_idx)
            src_shading = src_shading_dir + '/{:05d}.png'.format(test_idx)
            src_depth_gap = src_rec_dir + '/{:05d}.bmp'.format(test_idx)
        elif data_type is '3':
            src_bgra = src_frame_dir + '/{:05d}.png'.format(test_idx)
            src_depth_gt = src_gt_dir + '/{:05d}.bmp'.format(test_idx)
            src_shading = src_shading_dir + '/{:05d}.png'.format(test_idx)
            src_depth_gap = src_rec_dir + '/{:05d}.bmp'.format(test_idx)

        # read images
        bgr = cv2.imread(src_bgra, -1) / 255.
        bgr = bgr[:1200, :1200, :] 
        depth_img_gap = cv2.imread(src_depth_gap, -1)
        depth_img_gap = depth_img_gap[:1200, :1200, :]
        depth_gap = depth_tools.unpack_bmp_bgra_to_float(depth_img_gap)

        depth_img_gt = cv2.imread(src_depth_gt, -1)
        depth_img_gt = depth_img_gt[:1200, :1200, :]
        depth_gt = depth_tools.unpack_bmp_bgra_to_float(depth_img_gt)
        img_shape = bgr.shape[:2]

        shading_bgr = cv2.imread(src_shading, 1)
        shading_bgr = shading_bgr[:1200, :1200, :]
        shading_gray = cv2.imread(src_shading, 0) # GrayScale
        shading_gray = shading_gray[:1200, :1200]
        shading = shading_gray

        is_shading_available = shading > 0
        mask_shading = is_shading_available * 1.0
        depth_gap *= mask_shading

        if is_shading_norm: # shading norm : mean 0, var 1
            is_shading_available = shading > 8.0
            mask_shading = is_shading_available * 1.0
            mean_shading = np.sum(shading*mask_shading) / np.sum(mask_shading)
            var_shading = np.sum(np.square((shading - mean_shading)*mask_shading)) / np.sum(mask_shading)
            std_shading = np.sqrt(var_shading)
            shading = (shading - mean_shading) / std_shading
        else:
            shading = shading / 255.


        depth_thre = depth_threshold

        if is_input_coord:
            coord_x = np.linspace(0, 1, img_size)
            coord_y = np.linspace(0, 1, img_size)
            grid_x, grid_y = np.meshgrid(coord_x, coord_y)

        # merge bgr + depth_gap
        if is_input_frame:
            if is_input_depth:
                bgrd = np.dstack([shading[:, :], depth_gap, bgr[:, :, 0]])
                if is_input_coord:
                    bgrd = np.dstack([shading[:, :], depth_gap, bgr[:, :, 0], grid_x, grid_y])
            else:
                bgrd = np.dstack([shading[:, :], bgr[:, :, 0]])
        else:
            bgrd = np.dstack([shading[:, :], depth_gap])

        # clip batches
        b_top, b_left = batch_tl
        b_h, b_w = batch_shape
        top_coords = range(b_top, img_shape[0], b_h)
        left_coords = range(b_left, img_shape[1], b_w)

        # add test data
        x_test = []
        for top, left in product(top_coords, left_coords):

            def clip_batch(img, top_left, size):
                t, l, h, w = *top_left, *size
                return img[t:t + h, l:l + w]

            batch_test = clip_batch(bgrd, (top, left), batch_shape)
            
            if is_input_depth or is_input_frame:
                x_test.append(batch_test)
            else:
                x_test.append(batch_test[:, :, 0].reshape((*batch_shape, 1)))

        # predict
        x_test = np.array(x_test)[:]
        predict = model.predict(x_test, batch_size=1)  # w/o denormalization
        decode_img = predict[0][:, :, 0:2]

        # training types
        is_gt_available = depth_gt > depth_thre
        is_gap_unavailable = depth_gap < depth_thre

        is_depth_close = np.logical_and(
            np.abs(depth_gap - depth_gt) < difference_threshold,
            is_gt_available)

        is_to_interpolate = np.logical_and(is_gt_available, is_gap_unavailable)
        train_segment = np.zeros(decode_img.shape[:2])
        train_segment[is_depth_close] = 1
        train_segment[is_to_interpolate] = 2

        # mask = is_gt_available * 1.0 # GT
        mask = is_depth_close * 1.0 # no-complement
        # mask = is_train_area * 1.0 # complement

        # delete mask edge
        if is_edge_crop:
            # kernel = np.ones((mask_edge_size, mask_edge_size), np.uint8)
            # mask = cv2.erode(mask, kernel, iterations=1)

            edge_size = mask_edge_size
            mask_filter = np.zeros_like(mask)
            for edge in range(1, edge_size):
                edge_2 = edge * 2
                mask_filter[edge: b_h - edge, edge: b_w - edge] = mask[: b_h - edge_2, edge: b_w - edge]
                mask *= mask_filter
                mask_filter[edge: b_h - edge, edge: b_w - edge] = mask[edge: b_h - edge, edge_2: ]
                mask *= mask_filter
                mask_filter[edge: b_h - edge, edge: b_w - edge] = mask[edge_2: , edge: b_w - edge]
                mask *= mask_filter
                mask_filter[edge: b_h - edge, edge: b_w - edge] = mask[edge: b_h - edge, : b_w - edge_2]
                mask *= mask_filter

                for i in range(2):
                    for j in range(2):
                        mask_filter[
                            edge * i: b_h - edge * (1 - i), edge * j: b_w - edge * (1 - j)
                            ] = mask[
                                edge * (1 - i): b_h - edge * i, edge * (1 - j): b_h - edge * j
                                ]
                        mask *= mask_filter


        mask_gt = is_gt_available * 1.0

        mask_length = np.sum(mask)

        # cv2.imwrite(predict_dir + '/mask-{:05d}.png'.format(test_idx),
        #             (mask * 255).astype(np.uint8))

        predict_diff = decode_img[:, :, 0].copy()

        if is_pred_smooth:
            predict_depth = common_tools.gaussian_filter(predict_diff, 2, 0.002)

        depth_gt_masked = depth_gt * mask
        gt_diff = (depth_gt - depth_gap) * mask
        predict_diff_masked = predict_diff * mask

        # predict normalization
        if is_predict_norm_local:
            patch_size = norm_patch_size

            if is_norm_local_pix:
                p = norm_patch_size
                new_pred_diff = np.zeros_like(predict_diff_masked)
                new_mask = mask.copy()
                for i in range(p + 1, batch_shape[0] - p - 1):
                    for j in range(p + 1, batch_shape[1] - p - 1):
                        if not mask[i, j]:
                            new_pred_diff[i, j] = 0
                            continue
                        local_mask = mask[i-p:i+p+1, j-p:j+p+1]
                        local_gt_diff = gt_diff[i-p:i+p+1, j-p:j+p+1]
                        local_pred = predict_diff_masked[i-p:i+p+1, j-p:j+p+1]
                        local_mask_len = np.sum(local_mask)
                        patch_len = (p*2 + 1) ** 2
                        if local_mask_len < patch_len*patch_rate/100:
                            new_pred_diff[i, j] = 0
                            new_mask[i, j] = False
                            continue
                        local_mean_gt = np.sum(local_gt_diff) / local_mask_len
                        local_mean_pred = np.sum(local_pred) / local_mask_len
                        local_sd_gt = np.sqrt(np.sum(np.square(local_gt_diff - local_mean_gt)) / local_mask_len)
                        local_sd_pred = np.sqrt(np.sum(np.square(local_pred - local_mean_pred)) / local_mask_len)
                        if is_fix_inv_local:
                            new_local_gt = local_gt_diff - local_mean_gt
                            new_local_pred = (local_pred - local_mean_pred) * (local_sd_gt / local_sd_pred)
                            new_local_pred_inv = new_local_pred.copy() * -1
                            new_local_err = np.sqrt(np.sum(np.square(new_local_gt - new_local_pred)))
                            new_local_err_inv = np.sqrt(np.sum(np.square(new_local_gt - new_local_pred_inv)))
                            if new_local_err < new_local_err_inv:
                                new_pred_diff[i, j] = new_local_pred[p, p] + local_mean_gt
                            else:
                                new_pred_diff[i, j] = new_local_pred_inv[p, p] + local_mean_gt
                        else:
                            new_pred_diff[i, j] = (predict_diff[i, j] - local_mean_pred) * (local_sd_gt / local_sd_pred) + local_mean_gt
                predict_diff = new_pred_diff
                mask = new_mask
                mask[:p, :] = False
                mask[batch_shape[0] - p:, :] = False
                mask[:, :p] = False
                mask[:, batch_shape[1] - p:] = False
                mask *= 1.0
                mask_length = np.sum(mask)
            else:
                for i in range(batch_shape[0] // p):
                    for j in range(batch_shape[1] // p):
                        local_mask = mask[p*i:p*(i+1), p*j:p*(j+1)]
                        local_gt_diff = gt_diff[p*i:p*(i+1), p*j:p*(j+1)]
                        local_pred = predict_diff_masked[p*i:p*(i+1), p*j:p*(j+1)]
                        local_mask_len = np.sum(local_mask)
                        if local_mask_len < 10:
                            predict_diff[p*i:p*(i+1), p*j:p*(j+1)] = 0
                            continue
                        local_mean_gt = np.sum(local_gt_diff) / local_mask_len
                        local_mean_pred = np.sum(local_pred) / local_mask_len
                        local_sd_gt = np.sqrt(np.sum(np.square(local_gt_diff - local_mean_gt)) / local_mask_len)
                        local_sd_pred = np.sqrt(np.sum(np.square(local_pred - local_mean_pred)) / local_mask_len)
                        predict_diff[p*i:p*(i+1), p*j:p*(j+1)] = (local_pred - local_mean_pred) * (local_sd_gt / local_sd_pred) + local_mean_gt
            
        
        elif is_predict_norm:
            mean_gt = np.sum(gt_diff) / mask_length
            mean_predict = np.sum(predict_diff_masked) / mask_length
            gt_diff -= mean_gt
            predict_diff -= mean_predict
            out_diff_R = predict_diff.copy() # save diff
            sd_gt = np.sqrt(np.sum(np.square(gt_diff * mask)) / mask_length)
            sd_predict = np.sqrt(np.sum(np.square(predict_diff * mask)) / mask_length)
            predict_diff *= sd_gt / sd_predict
            predict_diff += mean_gt

        ###############
        depth_gt_masked = depth_gt * mask
        gt_diff = (depth_gt - depth_gap) * mask
        predict_diff_masked = predict_diff * mask

        # reverse predict
        if is_pred_reverse:
            predict_depth = predict_diff * -1.0 # reverse

            # difference learn
            predict_depth += depth_gap
            predict_masked = predict_depth * mask

            # ajust bias, calc error
            if is_pred_ajust:
                mean_depth_gt = np.sum(depth_gt_masked) / mask_length
                mean_depth_pred = np.sum(predict_masked) / mask_length
                predict_depth += mean_depth_gt - mean_depth_pred
                predict_masked += mean_depth_gt - mean_depth_pred

            # error
            depth_err_abs_R = np.abs(depth_gt - depth_gap)
            depth_err_sqr_R = np.square(depth_gt - depth_gap)
            if is_pred_ajust:
                predict_err_abs_R = np.abs(depth_gt - predict_depth)
                predict_err_sqr_R = np.square(depth_gt - predict_depth)
            else:
                predict_err_abs_R = np.abs(depth_gt - predict_depth)
                predict_err_sqr_R = np.square(depth_gt - predict_depth)

            # error image
            depth_err_R = depth_err_abs_R
            predict_err_R = predict_err_abs_R
            predict_err_masked_R = predict_err_R * mask
            # Mean Absolute Error
            predict_MAE_R = np.sum(predict_err_abs_R * mask) / mask_length
            depth_MAE_R = np.sum(depth_err_abs_R * mask) / mask_length
            # Mean Squared Error
            predict_MSE_R = np.sum(predict_err_sqr_R * mask) / mask_length
            depth_MSE_R = np.sum(depth_err_sqr_R * mask) / mask_length
            # Root Mean Square Error
            predict_RMSE_R = np.sqrt(predict_MSE_R)
            depth_RMSE_R = np.sqrt(depth_MSE_R)
            #################################################################

        predict_depth = predict_diff
        # difference learn
        predict_depth += depth_gap
        predict_masked = predict_depth * mask

        # ajust bias, calc error
        if is_pred_ajust:
            mean_depth_gt = np.sum(depth_gt_masked) / mask_length
            mean_depth_pred = np.sum(predict_masked) / mask_length
            predict_depth += mean_depth_gt - mean_depth_pred
            predict_masked += mean_depth_gt - mean_depth_pred

        # error
        depth_err_abs = np.abs(depth_gt - depth_gap)
        depth_err_sqr = np.square(depth_gt - depth_gap)
        depth_err_diff = depth_gt - depth_gap
        if is_pred_ajust:
            predict_err_abs = np.abs(depth_gt - predict_depth)
            predict_err_sqr = np.square(depth_gt - predict_depth)
        else:
            predict_err_abs = np.abs(depth_gt - predict_depth)
            predict_err_sqr = np.square(depth_gt - predict_depth)

        # error image ##################################################
        depth_err = depth_err_abs
        predict_err = predict_err_abs
        predict_err_masked = predict_err * mask

        # depth_err = depth_err_diff
        # predict_err_masked = (depth_gt - predict_depth) * mask
        #################################################################

        # Mean Absolute Error
        predict_MAE = np.sum(predict_err_abs * mask) / mask_length
        depth_MAE = np.sum(depth_err_abs * mask) / mask_length
        # Mean Squared Error
        predict_MSE = np.sum(predict_err_sqr * mask) / mask_length
        depth_MSE = np.sum(depth_err_sqr * mask) / mask_length
        # Root Mean Square Error
        predict_RMSE = np.sqrt(predict_MSE)
        depth_RMSE = np.sqrt(depth_MSE)

        # SD
        sd_ae_depth = np.sqrt(np.sum(np.square(depth_err_abs - depth_MAE) * mask) / mask_length)
        sd_ae_pred = np.sqrt(np.sum(np.square(predict_err_abs - predict_MAE) * mask) / mask_length)
        sd_rse_depth = np.sqrt(np.sum(np.square(np.sqrt(depth_err_sqr) - depth_RMSE) * mask) / mask_length)
        sd_rse_pred = np.sqrt(np.sum(np.square(np.sqrt(predict_err_sqr) - predict_RMSE) * mask) / mask_length)

        if is_pred_pix_reverse: # Inverse on Pix
            if is_reverse_threshold: # Inverse by Threshold
                predict_err_abs = np.where(np.logical_and(predict_err_abs > r_thre, predict_err_abs > predict_err_abs_R), 
                                            predict_err_abs_R, predict_err_abs)
                predict_err_sqr = np.where(np.logical_and(predict_err_sqr > r_thre**2, predict_err_sqr > predict_err_sqr_R), 
                                            predict_err_sqr_R, predict_err_sqr)
            else:
                predict_err_abs = np.where(predict_err_abs < predict_err_abs_R, predict_err_abs, predict_err_abs_R)
                predict_err_sqr = np.where(predict_err_sqr < predict_err_sqr_R, predict_err_sqr, predict_err_sqr_R)
            predict_err = predict_err_abs
            predict_err_masked = predict_err * mask
            predict_MAE = np.sum(predict_err_abs * mask) / mask_length
            predict_MSE = np.sum(predict_err_sqr * mask) / mask_length
            predict_RMSE = np.sqrt(predict_MSE)
        elif is_pred_patch_inverse: # Inverse on Patch
            p = norm_patch_size
            for i in range(batch_shape[0] // p):
                for j in range(batch_shape[1] // p):
                    err_abs_patch = predict_err_abs[p*i:p*(i+1), p*j:p*(j+1)]
                    err_abs_patch_R = predict_err_abs_R[p*i:p*(i+1), p*j:p*(j+1)]
                    err_sqr_patch = predict_err_sqr[p*i:p*(i+1), p*j:p*(j+1)]
                    err_sqr_patch_R = predict_err_sqr_R[p*i:p*(i+1), p*j:p*(j+1)]
                    if np.sum(err_sqr_patch) < np.sum(err_abs_patch_R):
                        predict_err_abs[p*i:p*(i+1), p*j:p*(j+1)] = err_abs_patch
                        predict_err_sqr[p*i:p*(i+1), p*j:p*(j+1)] = err_sqr_patch
                    else:
                        predict_err_abs[p*i:p*(i+1), p*j:p*(j+1)] = err_abs_patch_R
                        predict_err_sqr[p*i:p*(i+1), p*j:p*(j+1)] = err_sqr_patch_R

        elif is_pred_reverse:
            if predict_RMSE > predict_RMSE_R:
                depth_err = depth_err_R
                predict_err = predict_err_R
                predict_err_masked = predict_err_masked_R
                predict_MAE = predict_MAE_R
                depth_MAE = depth_MAE_R
                predict_MSE = predict_MSE_R
                depth_MSE = depth_MSE_R
                predict_RMSE = predict_RMSE_R
                depth_RMSE = depth_RMSE_R
                out_diff = out_diff_R

        # output difference
        if is_save_diff:
            net_out_dir = predict_dir + '/net_output/'
            os.makedirs(net_out_dir, exist_ok=True)
            if test_idx in save_img_range:
                np.save(net_out_dir + '{:05d}.npy'.format(test_idx), out_diff)
                out_diff_depth = out_diff + 1
                xyz_out_diff = depth_tools.convert_depth_to_coords(out_diff_depth, cam_params)
                depth_tools.dump_ply(net_out_dir + '{:05d}.ply'.format(test_idx), xyz_out_diff.reshape(-1, 3).tolist())


        err_strings += str(test_idx)
        if test_idx in test_range:
        # if test_idx not in train_range:
            err_strings += ',test,'
        else:
            err_strings += ',train,'
        for string in [depth_MAE, predict_MAE,depth_RMSE, predict_RMSE, sd_ae_depth, sd_ae_pred, sd_rse_depth, sd_rse_pred]:
            err_strings += str(string) + ','
        err_strings.rstrip(',')
        err_strings = err_strings[:-1] + '\n'

        predict_loss = predict_MAE
        depth_loss = depth_MAE

        # save predicted depth
        if is_save_depth_img:
            if test_idx in save_img_range:
                predict_bmp = depth_tools.pack_float_to_bmp_bgra(predict_masked)
                cv2.imwrite(predict_dir + '/predict-{:03d}.bmp'.format(test_idx), predict_bmp)

        # save ply
        if is_save_ply:
            if test_idx in save_ply_range:
                if is_masked_ply:
                    xyz_predict_masked = depth_tools.convert_depth_to_coords(predict_masked, cam_params)
                    depth_tools.dump_ply(predict_dir + '/predict_masked-%03d.ply'%test_idx, xyz_predict_masked.reshape(-1, 3).tolist())
                else:
                    xyz_predict = depth_tools.convert_depth_to_coords(predict_depth, cam_params)
                    depth_tools.dump_ply(predict_dir + '/predict-%03d.ply'%test_idx, xyz_predict.reshape(-1, 3).tolist())
                
        # save fig
        # if test_idx in test_range:
        if test_idx in save_img_range:
            # layout
            fig = plt.figure(figsize=(7, 5))
            gs_master = GridSpec(nrows=2,
                                ncols=2,
                                height_ratios=[1, 1],
                                width_ratios=[3, 0.1])
            gs_1 = GridSpecFromSubplotSpec(nrows=1,
                                        ncols=3,
                                        subplot_spec=gs_master[0, 0],
                                        wspace=0.05,
                                        hspace=0)
            gs_2 = GridSpecFromSubplotSpec(nrows=1,
                                        ncols=3,
                                        subplot_spec=gs_master[1, 0],
                                        wspace=0.05,
                                        hspace=0)
            gs_3 = GridSpecFromSubplotSpec(nrows=2,
                                        ncols=1,
                                        subplot_spec=gs_master[0:1, 1])

            ax_enh0 = fig.add_subplot(gs_1[0, 0])
            ax_enh1 = fig.add_subplot(gs_1[0, 1])
            ax_enh2 = fig.add_subplot(gs_1[0, 2])

            ax_misc0 = fig.add_subplot(gs_2[0, 0])

            ax_err_gap = fig.add_subplot(gs_2[0, 1])
            ax_err_pred = fig.add_subplot(gs_2[0, 2])

            ax_cb0 = fig.add_subplot(gs_3[0, 0])
            ax_cb1 = fig.add_subplot(gs_3[1, 0])

            for ax in [
                    ax_enh0, ax_enh1, ax_enh2,
                    ax_misc0, ax_err_gap, ax_err_pred
            ]:
                ax.axis('off')

            # close up
            mean = np.sum(depth_gt_masked) / mask_length
            vmin_s, vmax_s = mean - vm_range, mean + vm_range

            ax_enh0.imshow(depth_gt_masked, cmap='jet', vmin=vmin_s, vmax=vmax_s)
            ax_enh1.imshow(depth_gap * mask, cmap='jet', vmin=vmin_s, vmax=vmax_s)
            ax_enh2.imshow(predict_masked, cmap='jet', vmin=vmin_s, vmax=vmax_s)

            ax_enh0.set_title('Ground Truth')
            ax_enh1.set_title('Low-res')
            ax_enh2.set_title('Ours')

            # misc
            ax_misc0.imshow(shading_bgr[:, :, ::-1])

            # error
            is_scale_err_mm = True
            if is_scale_err_mm:
                scale_err = 1000
            else:
                scale_err = 1

            vmin_e, vmax_e = 0, vm_e_range * scale_err
            ax_err_gap.imshow(depth_err * mask * scale_err, cmap='jet', vmin=vmin_e, vmax=vmax_e)
            ax_err_pred.imshow(predict_err_masked * scale_err, cmap='jet', vmin=vmin_e, vmax=vmax_e)

            # colorbar
            plt.tight_layout()
            fig.savefig(io.BytesIO())
            cb_offset = -0.05

            plt.colorbar(ScalarMappable(colors.Normalize(vmin=vmin_s, vmax=vmax_s),
                                        cmap='jet'),
                        cax=ax_cb0)
            im_pos, cb_pos = ax_enh2.get_position(), ax_cb1.get_position()
            ax_cb0.set_position([
                cb_pos.x0 + cb_offset, im_pos.y0, cb_pos.x1 - cb_pos.x0,
                im_pos.y1 - im_pos.y0
            ])
            ax_cb0.set_xlabel('                [m]')

            plt.colorbar(ScalarMappable(colors.Normalize(vmin=vmin_e, vmax=vmax_e),
                                        cmap='jet'),
                        cax=ax_cb1)
            im_pos, cb_pos = ax_err_pred.get_position(), ax_cb1.get_position()
            ax_cb1.set_position([
                cb_pos.x0 + cb_offset, im_pos.y0, cb_pos.x1 - cb_pos.x0,
                im_pos.y1 - im_pos.y0
            ])
            if is_scale_err_mm:
                ax_cb1.set_xlabel('                [mm]')
            else:
                ax_cb1.set_xlabel('                [m]')

            plt.savefig(predict_dir + '/result-{:03d}.png'.format(test_idx), dpi=300)
            plt.close()

    with open(predict_dir + '/error_compare.txt', mode='w') as f:
        f.write(err_strings)

    compare_error.compare_error(predict_dir + '/')

if __name__ == "__main__":
    main()
