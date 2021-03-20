import numpy as np
import cupy as cp
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from scipy.spatial.distance import euclidean
import time
from tqdm import tqdm

import depth_tools

root_dir = 'C:/Users/Kodai Tokieda/Desktop/cnn-depth_remote/'
local_dir = 'C:/Users/b19.tokieda/Desktop/cnn-depth_remote/local-dir/'

# cam_params = {
#     'focal_length': 0.0360735,
#     'pix_x': 1.25e-05,
#     'pix_y': 1.2298133469700845e-05,
#     'center_x': 826.974,
#     'center_y': 543.754
# }
cam_params = { # input_data_1217
    'focal_length': 0.036640125,
    'pix_x': 1.25e-05,
    'pix_y': 1.2303973256411377e-05,
    'center_x': 801.895,
    'center_y': 602.872
}

# def compare_error(dir_name, error='RMSE'):
#     df_err = pd.read_csv(dir_name + '/error_compare.txt')
#     df_train = df_err[df_err['type']=='train']
#     df_test = df_err[df_err['type']=='test']

#     train_index = df_train['index'].astype(str).values
#     test_index = df_test['index'].astype(str).values

#     train_depth = np.array(df_train['{} depth'.format(error)])
#     train_predict = np.array(df_train['{} predict'.format(error)])
#     test_depth = np.array(df_test['{} depth'.format(error)])
#     test_predict = np.array(df_test['{} predict'.format(error)])

#     train_depth_mean = np.mean(train_depth)
#     train_predict_mean = np.mean(train_predict)
#     test_depth_mean = np.mean(test_depth)
#     test_predict_mean = np.mean(test_predict)

#     train_depth = np.append(train_depth, train_depth_mean)
#     train_predict = np.append(train_predict, train_predict_mean)
#     test_depth = np.append(test_depth, test_depth_mean)
#     test_predict = np.append(test_predict, test_predict_mean)

#     train_index = np.array(range(1, len(train_index) + 1))
#     test_index = np.array(range(1, len(test_index) + 1))

#     train_index = np.append(train_index, 'Avg')
#     test_index = np.append(test_index, 'Avg')

#     def plot(datatype, label, depth, predict):
#         plt.figure()
#         width = 0.3
#         index = np.array(range(len(label)))
#         bar1 = plt.bar(index-width, depth, width=width, align='edge', tick_label=label, color='lightblue')
#         bar2 = plt.bar(index, predict, width=width, align='edge', tick_label=label, color='orange')
#         plt.title('Error Comparison')
#         plt.xlabel(datatype + ' data')
#         plt.ylabel('{} [m]'.format(error))
#         plt.legend((bar1[0], bar2[0]), ('depth', 'predict'))
#         plt.tick_params(labelsize=7)
#         plt.savefig(dir_name + '/err_cmp_%s.pdf'%datatype)

#     plot('Training', train_index, train_depth, train_predict)
#     plot('Test', test_index, test_depth, test_predict)

# def compare_errors(dir1, dir2, dir_name='.', dir1_name='predict 1', dir2_name='predict 2', error='RMSE', opt_name=''):

#     df1_err = pd.read_csv(dir1 + '/error_compare.txt')
#     df1_train = df1_err[df1_err['type']=='train']
#     df1_test = df1_err[df1_err['type']=='test']

#     df2_err = pd.read_csv(dir2 + '/error_compare.txt')
#     df2_train = df2_err[df2_err['type']=='train']
#     df2_test = df2_err[df2_err['type']=='test']

#     train_index = df1_train['index'].astype(str).values
#     test_index = df1_test['index'].astype(str).values

#     train_depth = np.array(df1_train['{} depth'.format(error)])
#     test_depth = np.array(df1_test['{} depth'.format(error)])

#     train1_predict = np.array(df1_train['{} predict'.format(error)])
#     test1_predict = np.array(df1_test['{} predict'.format(error)])

#     train2_predict = np.array(df2_train['{} predict'.format(error)])
#     test2_predict = np.array(df2_test['{} predict'.format(error)])

#     train_depth_mean = np.mean(train_depth)
#     test_depth_mean = np.mean(test_depth)

#     train1_predict_mean = np.mean(train1_predict)
#     test1_predict_mean = np.mean(test1_predict)

#     train2_predict_mean = np.mean(train2_predict)
#     test2_predict_mean = np.mean(test2_predict)

#     train_depth = np.append(train_depth, train_depth_mean)
#     test_depth = np.append(test_depth, test_depth_mean)

#     train1_predict = np.append(train1_predict, train1_predict_mean)
#     test1_predict = np.append(test1_predict, test1_predict_mean)

#     train2_predict = np.append(train2_predict, train2_predict_mean)
#     test2_predict = np.append(test2_predict, test2_predict_mean)

#     train_index = np.array(range(1, len(train_index) + 1))
#     test_index = np.array(range(1, len(test_index) + 1))

#     train_index = np.append(train_index, 'Avg')
#     test_index = np.append(test_index, 'Avg')

#     if opt_name is not None:
#         opt_name = '_' + opt_name

#     def plot(datatype, label, depth, predict1, predict2):
#         plt.figure()
#         width = 0.25
#         index = np.array(range(len(label)))
#         bar1 = plt.bar(index-width, depth, width=width, align='edge', tick_label=label, color='lightblue')
#         bar2 = plt.bar(index, predict1, width=width, align='edge', tick_label=label, color='orange')
#         bar3 = plt.bar(index+width, predict2, width=width, align='edge', tick_label=label, color='lightgreen')
#         plt.title('Error Comparison')
#         plt.xlabel(datatype + ' data')
#         plt.ylabel('{} [m]'.format(error))
#         plt.legend((bar1[0], bar2[0], bar3[0]), ('depth', dir1_name, dir2_name))
#         plt.tick_params(labelsize=7)
#         plt.savefig(dir_name + '/errs_cmp%s_%s.pdf'%(opt_name, datatype))

#     plot('Training', train_index, train_depth, train1_predict, train2_predict)
#     plot('Test', test_index, test_depth, test1_predict, test2_predict)

def kmeans(depth, distance):
    height, width = depth.shape
    new_depth = np.zeros_like(depth)

    goal = height * width
    cnt = 0
    minimum = 0.00001

    for i in range(height):
        for j in range(width):
            cnt += 1
            print(' %07d / %07d'%(cnt, goal), end='\r')

            if j < distance or j >= height - distance:
                continue
            if np.abs(depth[i, j]) < minimum:
                continue

            patch = depth[i, j-distance:j+distance+1]
            length = np.sum(np.where(np.abs(patch) > minimum, 1, 0))
            new_depth[i, j] = np.sum(patch) / length

    return new_depth

def gaussian_filter(image, distance, threshold=0.005):
    height, width = image.shape
    diameter = distance*2 + 1
    blur = cv2.GaussianBlur(image, (diameter, diameter), 0)

    # cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
    # blur = cp.asarray(blur)
    # img = cp.asarray(image)
    
    # masked_blur = cp.where(cp.abs(blur - img) < threshold, blur, 0)
    masked_blur = np.where(np.abs(blur - image) < threshold, blur, 0)

    depth_threshold = 0.2
    is_gt_available= image > depth_threshold
    mask = is_gt_available * 1.0
    # return cp.asnumpy(masked_blur * mask)
    return masked_blur * mask
    # return blur

def euclid_filter(image, distance, grad=True):
    new_img = np.zeros_like(image)
    height, width = image.shape
    diameter = distance*2 + 1
    eucfilter = np.zeros((diameter, diameter))
    for i in range(diameter):
        for j in range(diameter):
            euclid = euclidean((i, j), (distance, distance))
            if euclid <= distance:
                if grad:
                    if euclid == 0:
                        eucfilter[i, j] = 1
                    else:
                        eucfilter[i, j] = 1 / euclid
                else:
                    eucfilter[i, j] = 1

    goal = height * width
    cnt = 0
    minimum = 0.00001

    for i in range(height):
        if i < distance or i >= height - distance:
            continue

        for j in range(width):
            cnt += 1
            print(' %07d / %07d'%(cnt, goal), end='\r')

            if j < distance or j >= width - distance:
                continue
            if np.abs(image[i, j]) < minimum:
                continue

            patch = image[i-distance:i+distance+1, j-distance:j+distance+1] * eucfilter
            length = np.sum(np.where(np.abs(patch) > minimum, 1, 0))
            new_img[i, j] = np.sum(patch) / length
    print('\n')
    return new_img

def euclid_filter_cupy(image, distance, grad=True):
    cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)

    img = cp.asarray(image).astype(cp.float32)
    new_img = cp.zeros_like(img)
    height, width = img.shape
    diameter = distance*2 + 1
    eucfilter = cp.zeros((diameter, diameter), dtype=cp.float32)
    for i in range(diameter):
        for j in range(diameter):
            euclid = euclidean((i, j), (distance, distance))
            if euclid <= distance:
                if grad:
                    if euclid == 0:
                        eucfilter[i, j] = 1
                    else:
                        eucfilter[i, j] = 1 / euclid
                else:
                    eucfilter[i, j] = 1

    get_smooth_image = cp.ElementwiseKernel(
        in_params='raw float32 img, raw float32 eucfilter, uint16 height, uint16 width, uint16 distance, uint16 diameter',
        out_params='float32 output',
        preamble=\
        '''
        __device__ int get_x_idx(int i, int width) {
            return i % width;
        }
        __device__ int get_y_idx(int i, int height) {
            return i / height;
        }
        ''',
        operation=\
        '''
        int x = get_x_idx(i, width);
        int y = get_y_idx(i, height);
        float minimum = 0.00001;
        float sum = 0;
        float length = 0;
        if ( ((x >= distance) && (x < width - distance)) && ((y >= distance) && (y < height - distance)) && (img[i] > minimum) ) {
            for (int k=0; k<diameter; k++) {
                for (int l=0; l<diameter; l++) {
                    float pixel_img = img[i + (k-distance)*height + l - distance];
                    float pixel_filter = eucfilter[k*diameter + l];
                    if (pixel_img > minimum) {
                        sum += pixel_img * pixel_filter;
                        length += pixel_filter;
                    }
                }
            }
            output = sum / length;
        } else {
            output = 0;
        }
        ''',
        name='get_smooth_image'
    )
    get_smooth_image(img, eucfilter, height, width, distance, diameter, new_img)
    return cp.asnumpy(new_img)

def delete_mask(image, mask):
    zeros = np.zeros_like(image)
    mask = mask.reshape(mask.shape + (1,))
    return np.where(mask == 0, image, zeros)

def main():
    DIR = 'input_data_1217/'

    time_total_start = time.time()

    for idx in tqdm(range(56)):
        # print(idx)
        # time_start = time.time()

        depth_img_gt = cv2.imread(DIR + 'gt_masked/gt{:03d}.bmp'.format(idx), -1)
        # depth_img_gt = depth_img_gt[:1200, :1200, :]
        # cv2.imwrite(DIR + 'gt-original/gt{:03d}.bmp'.format(idx), depth_img_gt)
        depth_gt = depth_tools.unpack_bmp_bgra_to_float(depth_img_gt)

        # depth_img_rec = cv2.imread(DIR + 'rec/depth{:03d}.png'.format(idx), -1)
        # depth_rec = depth_tools.unpack_png_to_float(depth_img_rec)

        # gt = cv2.imread(DIR + 'gt-original/gt{:03d}.bmp'.format(idx), -1)
        # gt_mask = cv2.imread(DIR + 'gt_mask/mask{:03d}.bmp'.format(idx), -1)
        # gt_masked = delete_mask(gt, gt_mask)
        # cv2.imwrite(DIR + 'gt_masked/gt{:03d}.bmp'.format(idx), gt_masked)

        new_depth_gt = gaussian_filter(depth_gt, 4)

        new_depth_gt_img = depth_tools.pack_float_to_bmp_bgra(new_depth_gt)
        cv2.imwrite(DIR + 'gt/gt{:03d}.bmp'.format(idx), new_depth_gt_img)

        new_xyz_depth_gt = depth_tools.convert_depth_to_coords(new_depth_gt, cam_params)
        depth_tools.dump_ply(DIR + 'ply_gt/gt{:03d}.ply'.format(idx), new_xyz_depth_gt.reshape(-1, 3).tolist())

        # xyz_depth_rec = depth_tools.convert_depth_to_coords(depth_rec, cam_params)
        # depth_tools.dump_ply(DIR + 'tmp/depth{:03d}.ply'.format(idx), xyz_depth_rec.reshape(-1, 3).tolist())

        # time_end = time.time()
        # print('{:.2f} sec'.format(time_end - time_start))

    time_total_end = time.time()
    print('total {:.2f} sec'.format(time_total_end - time_total_start))

    # compare_error('output_augment=1/predict_1000_val')
    # compare_errors('.', 'output_augment=0/predict_1000_val', 'output_augment=1/predict_1000_val')
    # compare_errors('.', 'output_augment=0/predict_1000_val', 'output_augment=1/predict_1000_val', 'no-augment', 'augment')

if __name__ == '__main__':
    main()