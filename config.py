# SetUp
shape_img = (1200, 1200)
shape_patch = (120, 120)
# Threshold
thre_depth = 0.1
thre_diff = 0.01
# Directory
dir_root_data = '../data/'
dir_root_model = '../models/'
dir_root_output = '../outputs/'
# Input
is_input_depth = True
is_input_frame = True
is_input_coord = True
# Normalization
is_shade_norm = True # Shading Normalization
is_diff_norm = True # Difference Normalization


batch_wave1_100 = {
    'name': 'batch_wave1_100',
    'size_train': 46,
    'size_val': 20
}
batch_wave1_400 = {
    'name': 'batch_wave1_400',
    'size_train': 197,
    'size_val': 80
}
batch_wave1double_200 = {
    'name': 'batch_wave1-double_200',
    'size_train': 94,
    'size_val': 40
}
batch_wave1double_400 = {
    'name': 'batch_wave1-double_800',
    'size_train': 395,
    'size_val': 168
}

data_dict = {
    'batch_1wave': batch_wave1_100,
    'batch_1wave_4light': batch_wave1_400,
    'batch_1wave-double': batch_wave1double_200,
    'batch_1wave-double_4light': batch_wave1double_400
}


savefile_render = {
    'gt': 'gt/{:05d}.bmp',
    'depth': 'rec/{:05d}.bmp',
    'shade': 'shading/{:05d}.png',
    'proj': 'proj/{:05d}.png',
    'lowres': 'lres/{:05d}.png'
}
savefile_board = {
    'gt': 'gt/gt{:03d}.bmp',
    'depth': 'rec_ajusted/depth{:03d}.bmp',
    'shade': 'shading/shading{:03d}.bmp',
    'proj': 'frame/frame{:03d}.png',
    'lowres': 'lowres/depth{:03d}.bmp'
}
savefile_real = {
    'gt': 'gt/{:05d}.bmp',
    'depth': 'rec/{:05d}.bmp',
    'shade': 'shading/{:05d}.png',
    'proj': 'proj/{:05d}.png',
    'lowres': 'lres/{:05d}.png'
}

info_1wave = {
    'dir': dir_root_data + 'render_wave1_300/',
    'save_file': savefile_render,
    'data_size': 300,
    'range_train': range(200),
    'range_test': range(200, 300)
}
info_2wave = {
    'dir': dir_root_data + 'render_wave2_1100/',
    'save_file': savefile_render,
    'data_size': 1100,
    'range_train': range(1000),
    'range_test': range(1000, 1100)
}
info_board = {
    'dir': dir_root_data + 'board/',
    'save_file': savefile_board,
    'data_size': 68,
    'range_train': range(40),
    'range_test': range(40, 68)
}
info_real = {
    'dir': dir_root_data + 'real/',
    'save_file': savefile_real,
    'data_size': 19,
    'range_train': range(0),
    'range_test': range(19)
}
