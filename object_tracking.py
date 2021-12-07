import pyrealsense2 as rs
import numpy as np
import cv2
from copy import copy

color_dir = '../measure/data/color/'
depth_dir = '../measure/data/depth/'

data_dir ='analysed/' 

frame = 0
frame_limit = 500

##
red_data = []
yellow_data = []

# Real sense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 90)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
frames = pipeline.wait_for_frames()
depth_frame = frames.get_depth_frame()
depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

## Define 'HSV' value and 'color band'
#red 
hsv_red = [170, 190, 165]
red_upper = [50, 50, 50]
red_lower = [50, 50, 50]

#yellow
hsv_yellow = [27, 100, 100]
yellow_upper = [50, 35, 50]
yellow_lower = [4, 65, 40]

def Make_background():
    color = []
    depth = []
    n=0

    while (n < 150):
        # color
        color_image = np.load(color_dir + str(n) + ".npy")
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        color.append(gray_image)
        # depth
        depth_image = np.load(depth_dir + str(n) + ".npy")
        depth.append(depth_image)

        n = n + 1    
        continue

    # color
    color = np.array(color)
    color = color.mean(axis=0)
    color = color.astype(np.uint8)
    # depth
    depth = np.array(depth)
    depth = depth.mean(axis=0)
    depth = depth.astype(np.uint16)
    
    return color, depth, n

def Load_image(n):
    # Load color
    color_image = np.load(color_dir + str(n) + ".npy")
    color_time = np.load(color_dir + "ts.npy")
    time = color_time[n]
    
    # Load depth
    depth_time = np.load(depth_dir + "ts.npy")
    adjusted_time = time - 52.0
    depth_frame = (np.abs(depth_time - adjusted_time)).argmin()
    depth_image = np.load(depth_dir + str(depth_frame) + ".npy")

    return color_image, depth_image

def Color_difference(color, background):
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    difference = cv2.absdiff(gray, background)
    color[difference<6] = [0, 0, 0]

    return color

def HSV_filtering(color_difference):
    # Change image 'color' to 'hsv'
    hsv = cv2.cvtColor(color_difference, cv2.COLOR_BGR2HSV)
    H = hsv[:,:,0]
    S = hsv[:,:,1]
    V = hsv[:,:,2]
    
    # Red_filtering
    red_only = np.zeros(H.shape, dtype=np.uint8)
    red_only[(H>hsv_red[0]-red_lower[0]) & (H<hsv_red[0]+red_upper[0]) & (S>hsv_red[1]-red_lower[1]) & (S<hsv_red[1]+red_upper[1]) & (V>hsv_red[2]-red_lower[2]) & (V<hsv_red[2]+red_upper[2])] = 255
    # Yellow_filtering
    yellow_only = np.zeros(H.shape, dtype=np.uint8)
    yellow_only[(H>hsv_yellow[0]-yellow_lower[0]) & (H<hsv_yellow[0]+yellow_upper[0]) & (S>hsv_yellow[1]-yellow_lower[1]) & (S<hsv_yellow[1]+yellow_upper[1]) & (V>hsv_yellow[2]-yellow_lower[2]) & (V<hsv_yellow[2]+yellow_upper[2])] = 255


    red_only = cv2.erode(red_only, None, iterations=1)
    red_only = cv2.erode(red_only, None, iterations=3)
    
    yellow_only = cv2.dilate(yellow_only, None, iterations=1)
    yellow_only = cv2.dilate(yellow_only, None, iterations=3)

    ## Estimate red ball
    mu = cv2.moments(red_only, False)
    if mu['m00'] != 0:
        pyr, pxr = int(mu['m10']/mu['m00']), int(mu['m01']/mu['m00'])

    else:
        pyr, pxr = 0, 0 

    ## Estimate yellow ball
    mu = cv2.moments(yellow_only, False)
    if mu['m00'] != 0:
        pyy, pxy = int(mu['m10']/mu['m00']), int(mu['m01']/mu['m00'])

    else:
        pyy, pxy = 0, 0 
    
    estimated_point = [0]*4
    estimated_point[0] = pyr
    estimated_point[1] = pxr
    estimated_point[2] = pyy
    estimated_point[3] = pxy

    return estimated_point

def Trimming(depth, background_depth, estimated_point):
    pyr = estimated_point[0]
    pxr = estimated_point[1]
    pyy = estimated_point[2]
    pxy = estimated_point[3]

    depth_red_mask = copy(depth)
    depth_yellow_mask = copy(depth)
    background_red_mask = copy(background_depth)
    background_yellow_mask = copy(background_depth)

    if pyr !=0 or pxr !=0 or pyy != 0 or pxy != 0:
        red_mask = np.zeros(depth.shape, dtype=np.uint8)
        yellow_mask = np.zeros(depth.shape, dtype=np.uint8)
        cv2.circle(red_mask,(pyr, pxr), 20 ,(255, 255, 255), -1)
        cv2.circle(yellow_mask,(pyy, pxy), 30 ,(255, 255, 255), -1)

        depth_red_mask[red_mask==0] = [0]
        depth_yellow_mask[yellow_mask==0] = [0]
        background_red_mask[red_mask==0] = [0]
        background_yellow_mask[yellow_mask==0] = [0]

    if pyr == 0 and pxr == 0 and pyy == 0 and pxy == 0:
        depth_red_mask.fill(0)
        depth_yellow_mask.fill(0)
        background_red_mask.fill(0)
        background_yellow_mask.fill(0)

    trimmed = [0]*2
    trimmed[0] = depth_red_mask
    trimmed[1] = depth_yellow_mask
    
    background_trimmed = [0]*2
    trimmed[0] = background_red_mask
    trimmed[1] = background_yellow_mask
    
    return trimmed, background_trimmed

def Ball_detection(trimmed, background_trimmed, estimated_position):
    difference_red = cv2.absdiff(trimmed[0], background_trimmed[0])
    difference_yellow = cv2.absdiff(trimmed[1], background_trimmed[1])

    red_only = cv2.inRange(difference_red, 150, 2000)
    red_only = cv2.erode(red_only, None, iterations=1)
    red_only = cv2.dilate(red_only, None, iterations=4)
 
    yellow_only = cv2.inRange(difference_yellow, 100, 2000)
    yellow_only = cv2.erode(yellow_only, None, iterations=1)
    yellow_only = cv2.dilate(yellow_only, None, iterations=4)

    mu = cv2.moments(red_only, False)
    if mu['m00'] != 0:
        pyr, pxr = int(mu['m10']/mu['m00']), int(mu['m01']/mu['m00'])
    else:
        pyr, pxr= estimated_position[0], estimated_position[1]

    mu = cv2.moments(yellow_only, False)
    if mu['m00'] != 0:
        pyy, pxy= int(mu['m10']/mu['m00']), int(mu['m01']/mu['m00'])
    else:
        pyy, pxy = estimated_position[2], estimated_position[3]

    detected_point = [0]*4
    detected_point[0] = pyr
    detected_point[1] = pxr
    detected_point[2] = pyy
    detected_point[3] = pxy

    return detected_point

def Get_z_and_get_position(detected_point, red_data, yellow_data):
    pyr = detected_point[0]
    pxr = detected_point[1]
    pyy = detected_point[2]
    pxy = detected_point[3]

    if pyr != 0 and  pxr != 0 and pyy != 0 and pxy != 0:
        # red
        z_red = depth_image[pxr, pyr]*depth_scale
        position_red = rs.rs2_deproject_pixel_to_point(depth_intrin, [pxr,pyr], z_red)

        # yellow
        z_yellow = depth_image[pxy, pyy]*depth_scale
        position_yellow = rs.rs2_deproject_pixel_to_point(depth_intrin, [pxy,pyy], z_yellow)

    else:
        position_red = [0, 0, 0]
        position_yellow = [0, 0, 0]

    position_red = np.array(position_red)
    red_data = np.append(red_data, position_red)

    position_yellow = np.array(position_yellow)
    yellow_data = np.append(yellow_data, position_yellow)

    return red_data, yellow_data

def Save_position_data(red_data, yellow_data):
    red_data = np.array(red_data).reshape(frame+1-150,3)
    yellow_data = np.array(yellow_data).reshape(frame+1-150,3)

    np.save(data_dir + "red.npy", red_data)
    np.save(data_dir + "yellow.npy", yellow_data)
    
    print(red_data)
    print('Saved')
# Create background image
background_color, background_depth, frame = Make_background()

while (frame<frame_limit):
    
    color_image, depth_image = Load_image(frame)

    color_difference = Color_difference(color_image, background_color)
    
    estimated_point = HSV_filtering(color_difference)
    
    trimmed, background_trimmed = Trimming(depth_image, background_depth, estimated_point)

    detected_point = Ball_detection(trimmed, background_trimmed, estimated_point)

    red_data, yellow_data = Get_z_and_get_position(detected_point, red_data, yellow_data)

    Save_position_data(red_data, yellow_data)


