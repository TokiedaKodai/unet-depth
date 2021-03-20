import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

root_dir = '../output/output_archive/'
data_dir = root_dir + '200124/'

data_dir = '../output/'
# data_dir = '../output/archive/200203/'

output_dir = 'output_'
# pred_dir = '/predict_200'
pred_dir = '/predict_500'
# pred_dir = '/predict_500_fake'
# pred_dir = '/predict_1000'
save_dir = data_dir

def get_index_depth(dirname, index=[], type_name='test', error='RMSE'):
    df = pd.read_csv(dirname + '/error_compare.txt')
    df = df[df['type']==type_name]
    if len(index) is not 0:
        df = df.loc[index]
    index = df['index'].astype(str).values
    depth = np.array(df['{} depth'.format(error)])
    mean_depth = np.mean(depth)
    depth = np.append(depth, mean_depth)
    index = np.append(index, 'Avg')
    return index, depth

def get_predict(dirname, index=[], type_name='test', error='RMSE'):
    df = pd.read_csv(dirname + '/error_compare.txt')
    df = df[df['type']==type_name]
    if len(index) is not 0:
        df = df.loc[index]
    predict = np.array(df['{} predict'.format(error)])
    mean_predict = np.mean(predict)
    predict = np.append(predict, mean_predict)
    return predict

def get_list_dir(list_compares, data_dir, output_dir, pred_dir):
    list_dir = []
    for dir_name in list_compares:
        list_dir.append(data_dir + output_dir + dir_name + pred_dir)
    return list_dir

def get_list_pred(list_dir):
    list_pred = []
    for directory in list_dir:
        pred = get_predict(directory)
        list_pred.append(pred)
    return list_pred

def gen_graph(label, depth, list_pred, list_compares, comp_name, type_name='test', save_dir='./', error='RMSE'):
    list_color = ['blue', 'orange', 'lightgreen', 'lightblue', 'red']
    list_bar = []
    list_legend = ['depth']
    list_legend.extend(list_compares)

    if comp_name is not '':
        comp_name = '_' + comp_name
    
    idx = np.array(range(len(label)))
    width = 0.8 / len(list_legend)

    plt.figure()
    list_bar.append(plt.bar(idx-width, depth, width=width, align='edge', tick_label=label, color=list_color[0]))
    for i, pred in enumerate(list_pred):
        list_bar.append(plt.bar(idx+width*i, pred, width=width, align='edge', tick_label=label, color=list_color[i+1]))
    plt.legend(list_bar, list_legend)
    plt.title('Error Comparison')
    # plt.title('No-fake data learning')
    plt.xlabel(type_name + ' data')
    # plt.xlabel('Fake test data')
    plt.ylabel('{} [m]'.format(error))
    plt.tick_params(labelsize=6)
    # plt.savefig(save_dir + 'errs_cmp{}_{}.pdf'.format(comp_name, type_name))
    plt.savefig(save_dir + 'errs_cmp{}_{}.pdf'.format(comp_name, error))

def compare_error(dir_name, error='RMSE'):
    # for type_name in ['train', 'test']:
    for type_name in ['test']:
        label, depth = get_index_depth(dir_name, type_name=type_name, error=error)
        pred = get_predict(dir_name, type_name=type_name, error=error)
        # gen_graph(label, depth, [pred], ['predict'], comp_name='', type_name=type_name, save_dir=dir_name)
        gen_graph(label, depth, [pred], ['predict'], comp_name=type_name, type_name=type_name, save_dir=dir_name, error=error)

def compare_errors(list_compares, comp_name='', data_dir=data_dir, output_dir=output_dir, pred_dir=pred_dir):
    list_dir = get_list_dir(list_compares, data_dir, output_dir, pred_dir)
    label, depth = get_index_depth(list_dir[0])
    list_pred = get_list_pred(list_dir)
    # comp_name += '_fake'
    gen_graph(label, depth, list_pred, list_compares, comp_name, save_dir=save_dir)


def main():
    # # # list_compares = ['unet_drop=0', 'unet_drop=5', 'unet_drop=10', 'unet_drop=20']
    # list_compares = ['unet_no-aug', 'unet_drop-5', 'unet_drop-10', 'unet_drop-20']
    # compare_errors(list_compares, 'unet_drops')
    # # list_compares = ['resnet_drop=0', 'resnet_drop=5', 'resnet_drop=10', 'resnet_drop=20']
    # list_compares = ['resnet_no-aug', 'resnet_drop-5', 'resnet_drop-10', 'resnet_drop-20']
    # compare_errors(list_compares, 'resnet_drops')
    # # list_compares = ['dense-resnet_drop=0', 'dense-resnet_drop=5', 'dense-resnet_drop=10', 'dense-resnet_drop=20']
    # list_compares = ['dense-resnet_no-aug', 'dense-resnet_drop-5', 'dense-resnet_drop-10', 'dense-resnet_drop-20']
    # compare_errors(list_compares, 'dense-resnet_drops')

    # list_compares = ['unet_no-aug', 'unet_aug-no-zoom', 'unet_aug']
    # compare_errors(list_compares, 'unet_augs')
    # list_compares = ['resnet_no-aug', 'resnet_aug-no-zoom', 'resnet_aug']
    # compare_errors(list_compares, 'resnet_augs')
    # list_compares = ['dense-resnet_no-aug', 'dense-resnet_aug-no-zoom', 'dense-resnet_aug']
    # compare_errors(list_compares, 'dense-resnet_augs')

    # list_compares = ['unet_drop=0', 'resnet_drop=0', 'dense-resnet_drop=0']
    # list_compares = ['unet_drop-5', 'resnet_drop-5', 'dense-resnet_drop-5']
    # compare_errors(list_compares, 'nets_drop-5')
    # list_compares = ['unet_drop-10', 'resnet_drop-10', 'dense-resnet_drop-10']
    # compare_errors(list_compares, 'nets_drop-10')
    # list_compares = ['unet_drop-20', 'resnet_drop-20', 'dense-resnet_drop-20']
    # compare_errors(list_compares, 'nets_drop-20')

    # list_compares = ['unet_no-aug', 'resnet_no-aug', 'dense-resnet_no-aug']
    # compare_errors(list_compares, 'nets_no-aug')
    # list_compares = ['unet_aug-no-zoom', 'resnet_aug-no-zoom', 'dense-resnet_aug-no-zoom']
    # compare_errors(list_compares, 'nets_aug-no-zoom')
    # list_compares = ['unet_aug', 'resnet_aug', 'dense-resnet_aug']
    # compare_errors(list_compares, 'nets_aug')


    list_compares = ['resnet_drop-5_no-aug', 'resnet_drop-5_aug-no-zoom', 'resnet_drop-5_aug']
    compare_errors(list_compares, 'resnet_drop-5_augs')


    # dir1 = root_dir + '200122/output_aug/predict_1000/'
    # dir2 = root_dir + '200123/output_aug/predict_1000/'
    # index=[44, 45, 46, 47]
    # label, depth = get_index_depth(dir1, index)
    # pred1 = get_predict(dir1, index)
    # pred2 = get_predict(dir2, index)
    # list_pred = [pred1, pred2]
    # gen_graph(label, depth, list_pred, ['fake data learn', 'no-fake data learn'], 'FakeLearn')

if __name__ == '__main__':
        main()