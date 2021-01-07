import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

from keras.callbacks import CSVLogger, ModelCheckpoint

import network as NT
import train_tools as TT
import info as I

# Learning Parameters
num_epoch = 300
num_ch = 3
size_batch = 64
size_train = 70
size_val = 30
lr = 0.001
rate_dropout = 0.1
rate_val = 0.3
is_transfer_learn = False
is_transfer_encoder = False
monitor_loss = 'val_loss'
verbose = 1

'''
ARGV
1: Model Name
2: Data Type
3: Epoch num
4-6: parameter
'''
argv = sys.argv
_, name_model, data_type, num_epoch, is_aug_lumi = argv
data_type = int(data_type)
num_epoch = int(num_epoch)
is_aug_lumi = int(is_aug_lumi)

# Data
list_data = [
    'batch_1wave', # 0
    'batch_1wave_4light', # 1
    'batch_1wave-double', # 2
    'batch_1wave-double_4light', # 3
    ]

def main():
    # info_data = I.info_2wave
    # x_data, y_data = TT.LoadData(dir_data, range_data, info_data['save_file'])

    # Dir Model
    dir_model = I.dir_root_model + name_model + '/'
    os.makedirs(dir_model, exist_ok=True)

    # Resume
    try:
        df_log = pd.read_csv(dir_model + 'training.log')
        resume_from = 'auto'
        initial_epoch = int(df_log.tail(1).index.values) + 1
    except:
        resume_from = None
        initial_epoch = 0

    # Data
    name_data = list_data[data_type]
    info_data = I.data_dict[name_data]
    dir_data = I.dir_root_data + info_data['name']

    # Generator
    train_generator = TT.MiniBatchGenerator(
        dir_data + '/train', 
        info_data['size_train'], 
        size_train, 
        is_aug_lumi)
    val_generator = TT.MiniBatchGenerator(
        dir_data + '/val', 
        info_data['size_val'], 
        size_val)

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
    # model_save_cb = ModelCheckpoint(
    #     dir_model + 'model-{epoch:04d}.hdf5',
    #     period=save_period,
    #     save_weights_only=True
    # )
    csv_logger_cb = CSVLogger(
        dir_model + 'training.log',
        append=(resume_from is not None)
    )

    # Load Weight
    if resume_from is not None:
        # model_file = dir_model + 'model-final.hdf5'
        model_file = dir_model + 'model-best.hdf5'
        model.load_weights(model_file)

    # model.fit(
    #     x_data,
    #     y_data,
    #     epochs=num_epoch
    #     batch_size=size_batch,
    #     initial_epoch=initial_epoch,
    #     shuffle=True,
    #     validation_split=rate_val,
    #     callbacks=[model_save_cb, csv_logger_cb],
    #     verbose=verbose
    # )
    model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.batches_per_epoch,
        epochs=num_epoch,
        initial_epoch=initial_epoch,
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
    df_log = pd.read_csv(dir_model + 'training.log')
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