#!/usr/bin/python3

import sys
import os
sys.path.append('./')

config_dir='model_config_test'
output_dir = '/media/disk_2t/w266/out'
n_examples = None
num_train_epochs = 1

for config_file_name in os.listdir(config_dir):
    config_file_path = config_dir+'/'+config_file_name
    run_cmd = 'python3 train.py --do_train=True\
                --config=' + config_file_path + \
                ' --output_dir=' + "'%s'" % output_dir + \
                ' --num_train_epochs=' + str(num_train_epochs)
    os.system(run_cmd)
