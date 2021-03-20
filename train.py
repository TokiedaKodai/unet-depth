import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import argparse

from keras.callbacks import CSVLogger, ModelCheckpoint

import network as NT
import train_tools as TT
import config as cf

'''
ARGV
1: Model Name
2: Data Type
3: Epoch num
4-6: parameter
'''

# Parser
parser = argparse.ArgumentParser()
parser.add_argument('name', help='model name to use training and test')
parser.add_argument('data', type=int, help='[0 - 3]: data type')
parser.add_argument('epoch', type=int, help='end epoch num')
parser.add_argument('--exist', action='store_true', help='add, if pre-trained model exist')
parser.add_argument('--min_train', action='store_true', help='add to re-train from min train loss')
parser.add_argument('--min_val', action='store_true', help='add to re-train from min val loss')
parser.add_argument('--aug_lumi', action='store_true', help='add to augment lumination on training')
parser.add_argument('--aug_lumi_val', action='store_true', help='add to augment lumination on validation')
parser.add_argument('--batch', type=int, default=64, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--drop', type=float, default=0.1, help='dropout rate')
parser.add_argument('--val', type=float, default=0.3, help='validation data rate')
parser.add_argument('--verbose', type=int, default=1, help='[0 - 2]: progress bar')
args = parser.parse_args()

# Model Parameters
name_model = args.name
is_model_exist = args.exist
is_load_min_train = args.min_train
is_load_min_val = args.min_val
save_period = 1

# Data Parameters
data_type = args.data

# Training Parameters
num_epoch = args.epoch
size_batch = args.batch # Default 4
lr = args.lr # Default 0.001
rate_dropout = args.drop # Default 0.1
rate_val = args.val # Default 0.3
verbose = args.verbose # Default 1
num_ch = 2
size_train = 70
size_val = 30
is_transfer_learn = False
is_transfer_encoder = False
monitor_loss = 'val_loss'

# Augmentation Parameters
is_aug_lumi = args.aug_lumi
is_aug_lumi_val = args.aug_lumi_val

# Data
list_data = [
    'batch_1wave', # 0
    'batch_1wave_4light', # 1
    'batch_1wave-double', # 2
    'batch_1wave-double_4light', # 3
    ]

# Directory
dir_model = cf.dir_root_model + name_model + '/'
dir_save = dir_model + 'save/'
log_file = dir_model + 'training.log'
file_model = dir_save + '/model-%04d.hdf5'

def main():
    init_epoch = 0
    if is_model_exist:
        df_log = pd.read_csv(log_file)
        end_point = int(df_log.tail(1).index.values) + 1
        init_epoch = end_point
        load_epoch = end_point
        if is_load_min_val:
            df_loss = df_log['val_loss']
            load_epoch = df_loss.idxmin() + 1
        elif is_load_min_train:
            df_loss = df_log['loss']
            load_epoch = df_loss.idxmin() + 1

    os.makedirs(dir_model, exist_ok=True)
    os.makedirs(dir_save, exist_ok=True)

    # Data
    name_data = list_data[data_type]
    info_data = cf.data_dict[name_data]
    dir_data = cf.dir_root_data + info_data['name']

    # Generator
    train_generator = TT.MiniBatchGenerator(
        dir_data + '/train', 
        info_data['size_train'], 
        size_train, 
        is_aug_lumi)
    val_generator = TT.MiniBatchGenerator(
        dir_data + '/val', 
        info_data['size_val'], 
        size_val,
        is_aug_lumi=False,
        is_aug_lumi_val=is_aug_lumi_val)

    model = NT.BuildUnet(
        num_ch=num_ch,
        lr=lr,
        rate_dropout=rate_dropout,
        is_transfer_learn=is_transfer_learn,
        is_transfer_encoder=is_transfer_encoder
    )

    model_save_cb = ModelCheckpoint(
        dir_model + 'model-best.hdf5',
        monitor=monitor_loss,
        verbose=verbose,
        save_best_only=True,
        save_weights_only=True,
        mode='min',
        period=1
    )
    model_save_cb = ModelCheckpoint(
        dir_model + 'model-{epoch:04d}.hdf5',
        period=save_period,
        save_weights_only=True
    )
    csv_logger_cb = CSVLogger(log_file)

    # Load Weight
    if is_model_exist:
        model.load_weights(file_model%load_epoch)

    # model.fit(
    #     x_data,
    #     y_data,
    #     epochs=num_epoch
    #     batch_size=size_batch,
    #     initial_epoch=init_epoch,
    #     shuffle=True,
    #     validation_split=rate_val,
    #     callbacks=[model_save_cb, csv_logger_cb],
    #     verbose=verbose
    # )
    model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.batches_per_epoch,
        epochs=num_epoch,
        initial_epoch=init_epoch,
        shuffle=True,
        callbacks=[model_save_cb, csv_logger_cb],
        validation_data=val_generator,
        validation_steps=val_generator.batches_per_epoch,
        verbose=verbose,
        max_queue_size=2)
    model.save_weights(dir_model + 'model-final.hdf5')

    # Loss Graph
    dir_loss = dir_model + 'loss/'
    os.makedirs(dir_loss, exist_ok=True)
    df_log = pd.read_csv(log_file)
    epochs = df_log.index + 1
    train_loss = df_log['loss'].values
    val_loss = df_log['val_loss'].values
    plt.figure()
    plt.plot(epochs, train_loss, label='train')
    plt.plot(epochs, val_loss, label='val')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(dir_loss + 'loss_{}.pdf'.format(num_epoch))

if __name__ == "__main__":
    main()