from cubeio import *
import numpy as np
import predict
import matplotlib.pyplot as plt 
from matplotlib.figure import Figure
import matplotlib.cm as cm

if __name__ == '__main__':
    # # Load the cube
    # predicted_cube = cubeio.load('0982_3','predicted')
    # lat = predicted_cube.get("lat")
    # lon = predicted_cube.get("lon")
    # class_x1 = predicted_cube.get("class_x1")
    # class_x2 = predicted_cube.get("class_x2")
    # ratio = class_x1 / class_x2


    # plt.scatter(lon, lat, c=ratio, cmap='viridis',vmin=2.58, vmax = 2.59, s=1)
    # plt.colorbar()  # 添加颜色条
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title('pcolormesh示例')
    # plt.show() 

    # input("Press Enter to continue...")
    processed_cube = cubeio.load('0982_3','processed')
    print("!!!")
