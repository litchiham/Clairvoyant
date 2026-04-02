from cubeio import *
import numpy as np
import predict
import matplotlib.pyplot as plt 
from matplotlib.figure import Figure
import matplotlib.cm as cm
import torch
import numpy as np
from models import resnet_3c

if __name__ == '__main__':
    # Load the cube
    predicted_cube = cubeio.load('4238_4','predicted')
    lat = predicted_cube.lat
    lon = predicted_cube.lon
    reg = predicted_cube.reg

    reg[reg<=0]=np.nan
    plt.scatter(lon, lat, c=reg, cmap='viridis', s=3)
    plt.colorbar()  # 添加颜色条
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('pcolormesh示例')
    plt.show() 

    input("Press Enter to continue...")
    




    


