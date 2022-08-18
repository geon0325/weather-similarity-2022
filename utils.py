import math
import argparse
import numpy as np
import datetime
from collections import defaultdict

import torch
import PIL
import torchvision.transforms as transforms


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default='0', type=str, help='gpu number')
    parser.add_argument("--modeltype", default="cnn", type=str, help='type of model')
    parser.add_argument("--batch_size", default=128, type=int, help='batch size')
    parser.add_argument("--epochs", default=20, type=int, help='number of epochs')
    parser.add_argument("--learning_rate", default=1e-6, type=float, help='learning rate')
    parser.add_argument("--dim", default=256, type=int, help='embedding dimension')
    parser.add_argument("--batchtype", default="inc", type=str, help='{inc/dec/uni}-{inc/dec/uni}')
    #parser.add_argument("--negbatch", default="inc", type=str, help='inc/dec/uni')
    #parser.add_argument("--delta", default="inc", type=str, help='inc/dec/uni')
    parser.add_argument("--gamma", default=0.5, type=float, help='pos delta')
    parser.add_argument("--subimage", action='store_true')
    return parser.parse_args()

def generate_batches(data_size, batch_size, shuffle=True):
    data = np.arange(data_size)
    
    if shuffle:
        np.random.shuffle(data)
    
    batch_num = math.ceil(data_size / batch_size)
    batches = np.split(np.arange(batch_num * batch_size), batch_num)
    batches[-1] = batches[-1][:(data_size - batch_size * (batch_num - 1))]
    
    for i, batch in enumerate(batches):
        batches[i] = [data[j] for j in batch]
        
    return batches

def construct_batch(data_list, batch):
    tf = transforms.ToTensor()
    img_data = []
    for i in batch:
        file_name = data_list[i]
        try:
            img = tf(PIL.Image.open(file_name)).squeeze(0)
            img_data[0:0] = [img]
        except:
            continue
    img_data = torch.stack(img_data)
    return img_data

def get_date(filename):
    date = filename[:-4].split('_')[-1]
    year = int(date[:4])
    month = int(date[4:6])
    day = int(date[6:8])
    hour = int(date[8:10])
    minute = int(date[10:12])
    return datetime.datetime(year, month, day, hour, minute, 0)

def write_log(log_path, log_dic):
    with open(log_path, 'a') as f:
        for _key in log_dic:
            f.write(_key + '\t' + str(log_dic[_key]) + '\n')
        f.write('\n')
        
        