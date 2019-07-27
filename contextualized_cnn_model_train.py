#!/usr/bin/python3

import sys
import os
sys.path.append('./')

config_dir='model_config_test'
output_dir = '/media/disk_2t/w266/out'
epochs = 4

for config_file_name in os.listdir(config_dir):
    config_file_path = config_dir+'/'+config_file_name
    run_cmd = 'python3 train.py --do_train=True \
                --config=' + config_file_path + \
                ' --num_train_epochs=' + str(epochs) + \
                ' --output_dir=' + "'%s'" % output_dir
    os.system(run_cmd)
