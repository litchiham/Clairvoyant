
import os
os.environ["OMEGA_BIN_PATH"] = r"H:\Omega1"
os.environ["OMEGA_PY_PATH"] = r"G:\ANACONDA\envs\general\lib\site-packages\omegapy"
from omegapy import omega_data as od
# 打开 cube
cube = od.OMEGAdata("0022_1")

cube_corr = od.corr_therm_atm(cube, npool=1)

