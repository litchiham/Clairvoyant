import numpy as np
from real_data_2 import *
import os
import warnings
warnings.filterwarnings('ignore')
from scipy import signal
from scipy import interpolate


import cubeio as cio
import omegapy.omega_data as od
import omegapy.useful_functions as uf
import argparse
from config import *

import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data as Data
import torchvision.transforms as transforms
import torchvision.datasets as datasets 
from models import *
# import get_spectrum_data
import pandas as pd
from sklearn import metrics
import pickle



os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 默认参数
class Args:
    def __init__(self):
        self.batch_size = 64
        self.resume = 'model_best.pth.tar'
        self.name = 'SCANet'
        self.tensorboard = False
        self.data = 'Mn'
        self.lr = 0.1

args = Args()

class Predicted:
    '''存储和一个cube的预测结果'''
    
    def __init__(self, points_array=None):
        """
        初始化预测结果对象
        
        参数:
        points_array: numpy数组，形状为(n, 5)，每行代表一个点
        [lon, lat, classification_x1, classification_x2, reg]
        """
        if points_array is None:
            self.points = np.empty((0, 5))  # 创建空的5列数组
        else:
            self.points = np.array(points_array, dtype=float)
    
    def get(self, key):
        if self.points.size == 0:
            return np.array([])
            
        key_map = {
            'lon': 0,
            'lat': 1, 
            'class_x1': 2,
            'class_x2': 3,
            'reg': 4
        }
        
        if key not in key_map:
            raise ValueError(f"不支持的key: {key}")
            
        col_idx = key_map[key]
        return self.points[:, col_idx]
    
    # 一些辅助方法
    def append(self, point):
        self.points = np.append(self.points, [point], axis=0)
    def to_array(self):
        return self.points

class Predict:
    cube_names = [] #for example ['0006_0']
    best_acc = 0
    best_mcc = 0
    best_auroc = 0
    best_sen = 0
    best_spe = 0
    _base_py_path=''
    def __init__(self):
        self._base_py_path = config.py_path
        self._base_buffer_path = config.buffer_path
        cio.log('Process', 'Initialization', 'INFO')
    def predict_cube(self, cube_name):
        global args
        # if args.tensorboard: configure("runs/%s"%(args.name))
        
        # Data loading code
        kwargs = {'num_workers': 8, 'pin_memory': True}       
        #datasets
        cubeio = cio.CubeIO()
        cube = cubeio.load(cube_name=cube_name, type='processed')
        cube_rf = cube.cube_rf.reshape(-1, *(cube.cube_rf.shape[2:]))
        cube_lat = cube.lat.reshape(-1, *(cube.lat.shape[2:]))
        cube_lon = cube.lon.reshape(-1, *(cube.lon.shape[2:]))
        cube_lam = cube.lam
        
        #nor+baseline
        intensity1 = jxjz(cube_lam,nor(SG(cube_rf)))
        f2 = interpolate.interp1d(cube_lam,intensity1,kind='cubic')
        x_pred = np.linspace(0.865,2.385,num=305)
        y_pred = f2(x_pred)
        Myinput = np.zeros([cube_rf.shape[0], 3, 305], dtype=float)
        #original
        Myinput[i][0][:] = y_pred       
        #1st
        y_pred = np.diff(y_pred)         
        Myinput[i][1][1:] = y_pred
        #2st
        y_pred = np.diff(y_pred)  
        Myinput[i][2][1:-1] = y_pred

        # create model
        if args.data == 'Mn':
            cls = 2
        elif args.data == 'Mn_dx':
            cls = 3
        elif args.data == 'Mn_dl':    
            cls = 1
        else:
            print("unrecognized task!")
        
        model = SCANet(BasicBlock, [2, 2, 2, 2], num_classes=cls)
        
        # get the number of model parameters
        print('Number of model parameters: {}'.format(
            sum([p.data.nelement() for p in model.parameters()])))
        
        model = model.cuda()

        # optionally resume from a checkpoint
        dir = "runs/%s_%s"%(args.name,args.data)
        save_dir = os.path.join(dir,args.resume)

        if args.resume:
            if os.path.isfile(save_dir):
                checkpoint = torch.load(save_dir)
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' "
                    .format(args.resume))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

        cudnn.benchmark = True
        
        criterion = nn.BCEWithLogitsLoss().cuda()

        # evaluate on test set
        self. test(cube_rf, model, criterion)



    def test(self, target_data, model, criterion):
        """Perform validation on the validation set"""
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        incorrect = 0
        Corr = 0
        auroc = 0
        total_correct = 0
        total_num = 0


        # switch to evaluate mode
        model.eval()
        
        y_true = []
        y_pred = []
        lens_list = []
        lens_right = []
        lens_wrong = []
        l_r =0
        l_w =0
        a=0

        # 转换target_data为torch张量
        target_tensor = torch.from_numpy(target_data).float()
        
        # 加载all和all_label数据
        all = np.load('features/features_Mn.npz')
        all_label = np.load('features/labels_Mn.npz')
        all = torch.from_numpy(all['arr_0']).float()
        all_label = torch.from_numpy(all_label['arr_0']).float()
        
        # 移动到GPU
        if torch.cuda.is_available():
            target_tensor = target_tensor.cuda()
            all = all.cuda()
            all_label = all_label.cuda()

        
        
        end = time.time()
        batch_size = config.batch_size
        n_samples = target_tensor.size(0)
        # 分批处理数据（模拟原始函数的循环）
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            target_batch = target_tensor[start_idx:end_idx]
            
            with torch.no_grad():
                target_var = torch.autograd.Variable(target_batch)

            # compute output
            output = model.predict(target_var, all, all_label)
            
            # measure metrics
            output1 = output.cpu()

            #reg
            # output1 = np.maximum(output1.detach().numpy(),0)
            #class
            # output2 = np.argmax(output1.detach().numpy(), axis=1)
            
            # #save class probability
            # df1 = pd.DataFrame(output1.detach().numpy())
            # df1.to_excel('class_pro.xlsx',
                # index=False,
                # engine='openpyxl')

            y_pred.extend(output1.tolist())



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
        lr = args.lr * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 225))
        # log to TensorBoard
        if args.tensorboard:
            cio.log('predict',f'learning_rate, {lr}, {epoch} ', 'INFO')
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
if __name__ == '__main__':
    predict=Predict()
    predict.predict_cube('0982_3')
        
    
    