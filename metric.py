import math
import PIL
import torchvision.transforms as transforms
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import imageio
from skimage import io
import utils
import numpy as np

def calculate_dstat(list_1, list_2):
    list2dict_1 = defaultdict(int)
    list2dict_2 = defaultdict(int)
    
    for val in list_1:
        list2dict_1[val] += 1
    for val in list_2:
        list2dict_2[val] += 1
        
    sum_1 = sum(list(list2dict_1.values()))
    for k in list2dict_1:
        list2dict_1[k] = list2dict_1[k] / sum_1
    sum_2 = sum(list(list2dict_2.values()))
    for k in list2dict_2:
        list2dict_2[k] = list2dict_2[k] / sum_2
    
    keylist = list(set(list2dict_1.keys()).union(set(list2dict_2.keys())))
    keylist = sorted(keylist)
    
    cum_1, cum_2 = 0, 0
    dstat = 0
    for k in keylist:
        cum_1 += list2dict_1[k]
        cum_2 += list2dict_2[k]
        
#         assert cum_1 <= sum_1 and cum_2 <= sum_2, str(cum_1) + "_" + str(cum_2)
        
        if dstat < abs(cum_1 - cum_2):
            dstat = abs(cum_1 - cum_2)

    return dstat
    
def calculate_portion_similarity(image1, image2, rx, ry, cx, cy, N=24, B=20):
    dist_i = []
    dist_j = []
    
    for i in range(rx, ry):
        for j in range(cx, cy):
            v_i = image1[i][j]
            v_j = image2[i][j]
            
            dist_i.append((int)(v_i * B))
            dist_j.append((int)(v_j * B))
        
    d_stat = calculate_dstat(dist_i, dist_j)
    
    return 1.0 - d_stat

def calculate_image_similarity(rawimage1, rawimage2, N=24, B=20):
    image_height, image_width = rawimage1.shape
    
    
    h = math.ceil(image_height / N)
    w = math.ceil(image_width / N)
    
    th = h * N
    tw = w * N
    
    image1 = np.zeros((th, tw))
    image2 = np.zeros((th, tw))
    image1[:image_height, :image_width] = rawimage1[:,:]
    image2[:image_height, :image_width] = rawimage2[:,:]
    
    total_sim = 0
    for ridx in range(N):
        for cidx in range(N):
            rx, ry = h * ridx, h * (ridx + 1)
            cx, cy = w * cidx, w * (cidx + 1)
            
            total_sim += calculate_portion_similarity(image1, image2, rx, ry, cx, cy)
            
    return total_sim / (N * N)

def calculate_channel_similarity(imagepath1, imagepath2, N=24, B=20):
    tf = transforms.ToTensor()
    assert os.path.isfile(imagepath1)
    assert os.path.isfile(imagepath2)

    # image1 = plt.imread(imagepath1)
    # image2 = plt.imread(imagepath2)
    image1 = tf(PIL.Image.open(imagepath1)).squeeze(0)
    image2 = tf(PIL.Image.open(imagepath2)).squeeze(0)

    sim_d = calculate_image_similarity(image1, image2, N, B)
    
    return sim_d

def calculate_imageindex_similarity(image_index1, image_index2, video2image, index2image, N=24, B=20):
    agg = 0
    dividing = 0
    tf = transforms.ToTensor()
    for imagepath1, imagepath2 in zip(index2image[image_index1], index2image[image_index2]):
        dividing += 1
        assert os.path.isfile(imagepath1)
        assert os.path.isfile(imagepath2)

        # image1 = plt.imread(imagepath1)
        # image2 = plt.imread(imagepath2)
        image1 = tf(PIL.Image.open(imagepath1)).squeeze(0)
        image2 = tf(PIL.Image.open(imagepath2)).squeeze(0)

        sim_d = calculate_image_similarity(image1, image2, N, B)
        agg += sim_d
    
    return agg / dividing

def calculate_videoindex_similarity(video_index1, video_index2, video2image, index2image, N=24, B=20):
    agg = 0
    dividing = 0
#     print(video2image[video_index1])
#     print(video2image[video_index2])
    tf = transforms.ToTensor()
    
    for imageindex1, imageindex2 in zip(video2image[video_index1], video2image[video_index2]):
        for imagepath1, imagepath2 in zip(index2image[imageindex1], index2image[imageindex2]):
            dividing += 1
            assert os.path.isfile(imagepath1)
            assert os.path.isfile(imagepath2)
            
            # image1 = plt.imread(imagepath1)
            # image2 = plt.imread(imagepath2)
            image1 = tf(PIL.Image.open(imagepath1)).squeeze(0)
            image2 = tf(PIL.Image.open(imagepath2)).squeeze(0)
            
            sim_d = calculate_image_similarity(image1, image2, N, B)
            agg += sim_d
    
    return agg / dividing

'''
def calculate_video_torch_similarity(video_torch1, video_torch2, N=24, B=20):
    agg = 0
    dividing = 0
    
    for i in range(video_torch1.shape[0]):
        for j in range(video_torch1.shape[1]):
            dividing += 1
            image1 = video_torch1[i][j][:][:].numpy()
            image2 = video_torch2[i][j][:][:].numpy()
            
            sim_d = calculate_image_similarity(image1, image2, N, B)
            agg += sim_d
    
    return agg / dividing
'''
