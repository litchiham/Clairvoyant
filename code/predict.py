import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
import cubeio as cio
from config import *

from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from models import *
from scipy import signal
from scipy import interpolate

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 默认参数
class Args:
    def __init__(self):
        self.batch_size = 64
        self.resume = 'model_best.pth.tar'
        self.name = 'SCANet'
        self.tensorboard = False
        self.data = 'Mn_dl'
        self.lr = 0.1

args = Args()



class Predict:
    cube_names = [] #for example ['0006_0']
    best_acc = 0
    best_mcc = 0
    best_auroc = 0
    best_sen = 0
    best_spe = 0
    def __init__(self):
        cio.log('Process', 'Initialization', 'INFO')
    def predict_single(self, cube_name):
        global args
        # if args.tensorboard: configure("runs/%s"%(args.name))
        
        # Data loading code
        kwargs = {'num_workers': 8, 'pin_memory': True}       

        #skip if predicted
        if os.path.exists(f'{config.py_path}/predicted/{cube_name}_predicted.npz'):
            cio.log(source='Predict', message=f'skip because {cube_name} is already predicted', type='INFO')
            return

        #datasets
        cube = cio.cubeio.load(cube_name=cube_name, type='processed')
        cube_rf = cube.cube_rf.reshape(-1, *(cube.cube_rf.shape[2:])) # type: ignore
        
        cube_lam = cube.lam # type: ignore
        spectra_num = cube_rf.shape[0]
        Myinput = np.zeros([spectra_num, 3, 305], dtype=float)
        class_x1s = np.zeros([spectra_num], dtype=float)
        class_x2s = np.zeros([spectra_num], dtype=float)
        regs = np.zeros([spectra_num], dtype=float)
        lats = cube.lat.reshape(-1, *(cube.lat.shape[2:])) # type: ignore
        lons = cube.lon.reshape(-1, *(cube.lon.shape[2:])) # type: ignore
        for i in range(spectra_num):
            Myinput[i] = get3c(cube_rf[i], cube_lam)
            cio.log("Predict", f"{i/spectra_num*100:.2f}%", 'INFO', flush = True)

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
        y_pred = self. _test(Myinput, model, criterion)
        y_pred = np.array(y_pred)
        class_x1s = y_pred[:,0,0,0]
        class_x2s = y_pred[:,0,0,1]

        predicted = cio.Predicted()
        predicted.points = np.column_stack([lats, lons, class_x1s, class_x2s, regs])

        #save
        predicted_path=os.path.join(config.py_path, 'predicted', f'{cube_name}_predicted.npz')
        cio.save_Predicted(predicted_cube=predicted, filepath=predicted_path)
        return predicted
    
    def predict_cubes(self, cube_names, max_workers=1):
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.predict_single, cube_name) for cube_name in cube_names]
            for future in as_completed(futures):
                try:
                    result = future.result()
                    # print(f"Predicted for cube: {result}")
                except Exception as e:
                    print(f"Error occurred while predicting for cube: {e}")
                



    def _test(self, target_data, model, criterion):
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
            output_reg = np.maximum(output1.detach().numpy(),0)
            output_class = np.argmax(output1.detach().numpy(), axis=1)
            


            #reg
            # output1 = np.maximum(output1.detach().numpy(),0)
            #class
            # output2 = np.argmax(output1.detach().numpy(), axis=1)
            
            # #save class probability
            # df1 = pd.DataFrame(output1.detach().numpy())
            # df1.to_excel('class_pro.xlsx',
                # index=False,
                # engine='openpyxl')

            y_pred.extend(output1.detach().numpy())
        return y_pred



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

if __name__ == '__main__':
    config.log_level='INFO'
    predict=Predict()
    predicted = predict.predict_cubes(cube_names=['0982_3', '4238_4'], max_workers=2)

        
    
    