
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision.transforms as transforms

from scipy import signal
from scipy import interpolate

import cubeio as cio
from config import *


import numpy as np

import warnings
warnings.filterwarnings('ignore')
from scipy import signal

# 数据预处理
def SG(data, w=11, p=2):
    """
    :param data: raw spectrum data, shape (n_samples, n_features)
    :param w: int
    :param p: int
    :return: data after SG :(n_samples, n_features)
    """
    return signal.savgol_filter(data, w, p)

def nor(data):
    return (data-min(data))/(max(data)-min(data))

def jxjz(x,y_uniform):
    n = 5
    p0 = np.polyfit(x,y_uniform,n)#多项式拟合，返回多项式系数
    #print(p0)
    y_fit0 = np.polyval(p0,x) #计算拟合值
    # print(y_fit0)
    r0 = y_uniform-y_fit0
    dev0 = np.sqrt(np.sum((r0-np.mean(r0))**2)/len(r0)) #计算残差
    y_remove0 = y_uniform[y_uniform <= y_fit0] #峰值消除
    x_remove0 = x[np.where(y_uniform <= y_fit0)] #峰值消除
    
    i=0
    judge=1
    dev=[]
    while judge:
        p1 = np.polyfit(x_remove0, y_remove0, n)  # 多项式拟合，返回多项式系数
        y_fit1 = np.polyval(p1, x_remove0)  # 计算拟合值
        r1 = y_remove0 - y_fit1
        dev1 = np.sqrt(np.sum((r1 - np.mean(r1)) ** 2) / len(r1))  # 计算残差
        dev.append(dev1)
        if i == 0:
            judge = abs(dev[i] - dev0) / dev[i] > 0.05;
        else:
            judge = abs((dev[i] - dev[i-1]) / dev[i]) > 0.05; # 残差判断条件
        y_remove0[np.where(y_remove0 >= y_fit1)] = y_fit1[np.where(y_remove0 >= y_fit1)]; # 光谱重建
        i=i+1
    y_baseline=np.polyval(p1, x)  #基线
    y_baseline_correction=y_uniform-y_baseline  #基线校正后
    
    return y_baseline_correction

def get3c(intensity,wavelengths):
    Myinput = np.zeros([3, 305], dtype=float)
    y_pred = jxjz(wavelengths,nor(SG(intensity)))
    f2 = interpolate.interp1d(wavelengths,y_pred,kind='cubic')
    x_pred = np.linspace(0.865,2.385,num=305)
    y_pred = f2(x_pred)
    # original\n",
    Myinput[0][:] = y_pred

    #计算相邻差作为梯度gradients\n",
    y_pred = np.diff(y_pred)  
    Myinput[1][1:] = y_pred

    #计算相邻差作为二阶导\n",
    y_pred = np.diff(y_pred)  
    Myinput[2][1:-1] = y_pred
    
    return Myinput