import numpy as np
data = np.load(r"D:\Project\Clairvoyant-data\py\predicted\4238_4_processed.npy")
print(data)
data = data.reshape(-1,3)
import matplotlib.pyplot as plt
plt.scatter(x=data[:,1], y=data[:,2], c = data[:,0], cmap='viridis', s=3)
plt.colorbar()  # 添加颜色条
plt.xlabel('X')
plt.ylabel('Y')
plt.title('pcolormesh示例')
plt.show() 

input("Press Enter to continue...")