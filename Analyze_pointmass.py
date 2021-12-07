import numpy as np
import copy
from scipy.optimize import curve_fit

data_dir = 'analysed/'
data = np.load(data_dir + 'yellow.npy')

x_data = []
y_data = []
z_data = []
for f in range(data.shape[0]):
    x_data.append(data[f,0])
    y_data.append(data[f,1])
    z_data.append(data[f,2])

x = np.array(x_data)
z = np.array(z_data)

camera_hight = 0.76

y_data = y_data + camera_hight
y = np.empty((0,y_data.size))
y_data = np.array([y_data])
lavel = np.zeros(y_data.shape)
y = np.append(y,y_data,axis=0)
y = np.append(y,lavel,axis=0)

# x(t)とz(t)の運動関数を求める
fit_x = np.polyfit(time,x,4)
fit_z = np.polyfit(time,z,4)

# vy
v_y = []
for i in range(y[0].size-1):
    v_y_i = (y[0,i+1]-y[0,i])/(time[i+1]-time[i])
    v_y.append(v_y_i)

# Estimate bound section
VV = []
for j in range(y[0].size-2):
    vv = v_y[j]*v_y[j+1]
    VV.append(vv)

VV = np.array(VV)

bound_index = np.where(VV < -25)
bound_index = np.array(bound_index)
bound_count = bound_index.size

# detection 't'
if bound_index.size !=0:
    
    # 衝突区間い間れた範囲を反転
    for k in range(0,bound_index.size):
        for l in range(y[1].size):
            if l > bound_index[0,k] + 1:
                y[1,l]=k+1
    y_reverse = copy.copy(y)
    y_reverse[0,np.where(y[1]%2!=0)] = (-1)*y_reverse[0,np.where(y[1]%2!=0)]

    bound_time = []
    for m in range(0,bound_index.size):
        y_bound_section = []
        t_bound_section = []
        
        for n in range(bound_index[0,m]+1-5, bound_index[0,m]+1+5):
            y_bound_section.append(y_reverse[0,n])
            t = time[n]
            t_bound_section.append(t)
        
        y_res = np.polyfit(t_bound_section,y_bound_section,1)
        bound_time.append(-(y_res[1])/(y_res[0]))
        
    # 反発係数を求める
    def y_fitting_function_1(X,v_0):
        Y1 = -0.5*9.8*X**2 + v_y*X + y[0,0]
        return Y1

    y_section_1 = []
    t_section_1 = []
    for o1 in range(0,bound_index[0,0]):
        y_section_1.append(y[0,o1])
        t_section_1.append(time[o1])

    v_0,pcov = curve_fit(y_fitting_function_1, t_section_1, y_section_1)

    #v_1 = v_y[0]-9.8*bound_time[0]
    v_1 = v_0-9.8*bound_time[0]
    t_1 = bound_time[0]

    def y_fitting_function_2(X,v_2):
        Y2 = -0.5*9.8*(X - t_1)**2 + v_2*(X-t_1)
        return Y2
    
    y_section_2 = []
    t_section_2 = []
    for o2 in range(bound_index[0,0]+1,bound_index[0,1]+2):
        y_section_2.append(y[0,o2])
        t_section_2.append(time[o2])

    v_2,pcov = curve_fit(y_fitting_function_2,t_section_2,y_section_2)  
    e = -(v_2/v_1)


print('fit_x:',fit_x)
print('fit_z:',fit_z)
print('fit_y:',y[0,0],v_y[0])
print('e:',e)
print('bound_time',bound_time)
