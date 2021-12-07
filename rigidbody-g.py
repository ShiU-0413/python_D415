import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy.optimize
import functools
import math
from scipy.optimize import curve_fit

######
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.invert_xaxis()



data_dir = 'analysed/'
red_data = np.load(data_dir + 'red.npy')
yellow_data = np.load(data_dir + 'yellow.npr')
time = np.load(data_dir + 'time.npy')

x_red = []
y_red = []
z_red = []
x_yellow = []
y_yellow = []
z_yellow = []

for f in range(red_data.shape[0]):
    x_red.append(red_data[f,0])
    y_red.append(red_data[f,1])
    z_red.append(red_data[f,2])

    x_yellow.append(yellow_data[f,0])
    y_yellow.append(yellow_data[f,1])
    z_yellow.append(yellow_data[f,2])

gx = []
gy = []
gz = []

a = 0.2417
for g in range(np.array(x_red).size):
    rx = rx_data[i]
    ry = ry_data[i]
    rz = rz_data[i]
    yx = yx_data[i]
    yy = yy_data[i]
    yz = yz_data[i]

    x_g = (1-a)*x_red[g] + a*x_yellow[g]
    y_g = (1-a)*y_red[g] + a*y_yellow[g]
    z_g = (1-a)*z_red[g] + a*z_yellow[g]

    gx.append(x)
    gy.append(y)
    gz.append(z)

camera_hight = 0.76

gx = np.array(gx)
gy = np.array(gy) + camera_hight 
gz = np.array(gz)


fitx = np.polyfit(time,gx,1)
fitz = np.polyfit(time,gz,1)

y_0 = gy[0]
def detec_v0(X,v_0,y_0):
    Y = -0.5*9.8*X**2 + v_0*X + y_0
    return Y

vy_0,pcov = curve_fit(detec_v0,time,gy)

print('fit_x:',fitx)
print('fit_y:',np.array([y_0,vy_0]))
print('fit_z:',fitz)
