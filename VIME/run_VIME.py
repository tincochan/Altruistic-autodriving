import datetime
import os
import warnings
from datetime import timedelta
from parser import get_parser

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import seed_everything
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical

import dataset
from supervised_models import logit, mlp, xgb_model
from utils import *
from vime_self import vime_self
from vime_semi import vime_semi
from vime_utils import perf_metric

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  

# load data
parser = get_parser()
args = parser.parse_args()
print(args)

chosen_data = args.data
log_name = "%sData" % chosen_data[-1]
if chosen_data == 'real-n':
    data = dataset.Ndata(path='~/Custom-Semi-Supervised/data/ndata.csv')
elif chosen_data == 'real-m':
    data = dataset.Mdata(path='~/Custom-Semi-Supervised/data/mdata.csv')
elif chosen_data == 'real-t':
    data = dataset.Tdata(path='~/Custom-Semi-Supervised/data/tdata.csv')
elif chosen_data == 'real-c':
    data = dataset.Cdata(path='~/Custom-Semi-Supervised/data/cdata.csv')

# args
seed = args.seed
epochs = args.epoch
batch_size = args.batch_size
dim = args.dim
lr = args.lr
weight_decay = args.l2
initial_inspection_rate = args.initial_inspection_rate
train_begin = args.train_from 
test_begin = args.test_from
test_length = args.test_length
valid_length = args.valid_length
chosen_data = args.data
gpu_id = args.device

# Initial dataset split
seed_everything(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

assert initial_inspection_rate < 100, "for supervised setting, please set 99.9 instead of 100"
assert initial_inspection_rate > 0,   "initial_inspection_rate should be greater than 0"

# Initial dataset split
train_start_day = datetime.date(int(train_begin[:4]), int(train_begin[4:6]), int(train_begin[6:8]))
test_start_day = datetime.date(int(test_begin[:4]), int(test_begin[4:6]), int(test_begin[6:8]))
test_length = timedelta(days=test_length)    
test_end_day = test_start_day + test_length
valid_length = timedelta(days=valid_length)
valid_start_day = test_start_day - valid_length

# data
data.split(train_start_day, valid_start_day, test_start_day, test_end_day, valid_length, test_length, args)
data.featureEngineering()
  
# Hyper-parameters
p_m = 0.3
alpha = 2.0
K = 3
beta = 1.0

# Metric
metric = 'acc'

# MLP
mlp_parameters = dict()
mlp_parameters['hidden_dim'] = 100
mlp_parameters['epochs'] = 100
mlp_parameters['activation'] = 'relu'
mlp_parameters['batch_size'] = 100

# train
scaler = MinMaxScaler()
x_train = data.dftrainx_lab.values
x_train = scaler.fit_transform(x_train)
y_train = to_categorical(data.train_cls_label)

# unlab
x_unlab = data.dftrainx_unlab.values
x_unlab = scaler.transform(x_unlab)

# test
x_test = data.dftestx.values
x_test = scaler.transform(x_test)
y_test = to_categorical(data.test_cls_label)


# # Train VIME-Self
vime_self_parameters = dict()
vime_self_parameters['batch_size'] = 128
vime_self_parameters['epochs'] = 20
vime_self_encoder = vime_self(x_unlab, p_m, alpha, vime_self_parameters)
  
# Save encoder
if not os.path.exists('save_model'):
    os.makedirs('save_model')

file_name = './save_model/vime_model.h5'
  
vime_self_encoder.save(file_name)  
        
# Test VIME-Self
x_train_hat = vime_self_encoder.predict(x_train)
x_test_hat = vime_self_encoder.predict(x_test)

# testing 
y_test_hat = mlp(x_train_hat, y_train, x_test_hat, mlp_parameters)
results = perf_metric(metric, y_test, y_test_hat)
    
print('VIME-Self Performance: ' + str(results))

# Train VIME-Semi
vime_semi_parameters = dict()
vime_semi_parameters['hidden_dim'] = args.dim
vime_semi_parameters['batch_size'] = 128
vime_semi_parameters['iterations'] = 1000
y_test_hat = vime_semi(x_train, y_train, x_unlab, x_test, 
                       vime_semi_parameters, p_m, K, beta, file_name)

overall_f1,auc,pr, re, f, rev = metrics(y_test_hat[:,1], data.test_cls_label,data.test_reg_label,args)
