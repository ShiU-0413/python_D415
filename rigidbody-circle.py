import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy.optimize
import functools
import math

def plane(x, y, params):
    a = params[0]
    b = params[1]
    c = params[2]
    z = a*x + b*y + c
    return z

def error(params, points):
    result = 0
    for (x,y,z) in points:
        plane_z = plane(x, y, params)
        diff = abs(plane_z - z)
        result += diff**2
    return result

def cross(a, b):
    return [a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]]
#
data_dir = 'analysed/'
red_data = np.load(data_dir + 'red_npy')
yellow_data = np.load(data_dir + 'yellow.npy')
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

points = []
a = 0.2417
for g in range(x_red.size):
    x_g = (1-a)*x_red[g] + a*x_yellow[g]
    y_g = (1-a)*y_red[g] + a*y_yellow[g]
    z_g = (1-a)*z_red[g] + a*z_yellow[g]

    points.append((x_yellow - x_g, y_yellow - y_g, z_yellow - z_g))
 
fun = functools.partial(error, points=points)
params0 = [0, 0, 0]
res = scipy.optimize.minimize(fun, params0)

a = res.x[0]
b = res.x[1]
c = res.x[2]

point  = np.array([0.0, 0.0, c])
normal = np.array(cross([1,0,a], [0,1,b]))
###unit normal vector determination###
normal[0] = normal[0]/math.sqrt(normal[0]**2 + normal[1]**2 + normal[2])
normal[1] = normal[1]/math.sqrt(normal[0]**2 + normal[1]**2 + normal[2])
normal[2] = normal[2]/math.sqrt(normal[0]**2 + normal[1]**2 + normal[2])

#
pi_normal = math.atan(normal[1]/normal[0])
xt = normal[0]*math.cos(-pi_normal)-normal[1]*math.sin(-pi_normal)
yt = normal[0]*math.sin(-pi_normal)+normal[1]*math.cos(-pi_normal)
zt = normal[2]
theta_normal = math.atan(normal[0]/normal[2])

x_pi_rotation = np.empty(n)
y_pi_rotation = np.empty(n)
z_pi_rotation = np.empty(n)

x_theta_rotation = np.empty(n)
y_theta_rotation = np.empty(n)
z_theta_rotation = np.empty(n)

Pi = np.empty(n)
points_r = np.empty(0)

for i in range(n):
    x_pi_rotation[i] = points[i][0]*math.cos(-pi_normal)-points[i][1]*math.sin(-pi_normal)
    y_pi_rotation[i] = points[i][0]*math.sin(-pi_normal)+points[i][1]*math.cos(-pi_normal)
    z_pi_rotation[i] = points[i][2]

    x_theta_rotation[i] = x_r1[i]*math.cos(-theta_normal)+z_r1[i]*math.sin(-theta_normal)
    y_theta_rotation[i] = y_r1[i]
    z_theta_rotation[i] = -x_r1[i]*math.sin(-theta_normal)+z_r1[i]*math.cos(-theta_normal)
    points_r = np.append(points_r, [x_theta_rotation[i], y_theta_rotation[i], z_theta_rotation[i]])

    Pi[i] = math.acos(x_r[i]/math.sqrt(x_r[i]**2+y_r[i]**2))

    print("Pi=",Pi[i]*180/math.pi)


print("normal vector=", normal[0], normal[1], normal[2])

plt.show()

