#!/usr/bin/python3

import sys
import os
sys.path.append('./')

config_dir='model_configs/CNN'
training_file_path = './out/features/train_non_fine_tuned'
training_file_name = training_file_path.split('/')[3]
n_examples = 100
num_train_epochs = 1

for config_file_name in os.listdir(config_dir):
    config_file_path = config_dir+'/'+config_file_name
    run_cmd = 'python3 train.py --do_train=True\
                --config=' + config_file_path + \
                ' --tf_record=' + "'%s'" % training_file_name + \
                ' --num_train_examples=' + str(n_examples) + \
                ' --num_train_epochs=' + str(num_train_epochs)
    os.system(run_cmd)
