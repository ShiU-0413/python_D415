# ライブラリのインポート
import pyrealsense2 as rs
import numpy as np
import os

# データ保存ディレクトリパスの指定とディレクトリの作成
color_dir = ’data/color/’
depth_dir = ’data/depth/’
os.makedirs(color_dir)
os.makedirs(depth_dir)

# 測定フレーム数の設定
frame_limit = 500
# データを格納するリストの作成
color_data = [0]*(frame_limit+1)
depth_data = [0]*(frame_limit+1)
color_ts = []
depth_ts = []

# Realsense の起動・設定
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 90)
pipeline = rs.pipeline()
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_sensor.set_option(rs.option.laser_power, 300)
depth_scale = depth_sensor.get_depth_scale()

# 関数定義
# 画像を取得し，メモリに画像を保存するための関数
def Measure(n):
    # 画像の取得
　　frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    depth_time = depth_frame.get_timestamp()
    depth_image = np.array(depth_frame.get_data())
    color_frame = frames.get_color_frame()
    color_time = color_frame.get_timestamp()
    color_image = np.array(color_frame.get_data())
    # リストに取得画像をリストに格納
    depth_data[n] = depth_image
    depth_ts.append(depth_time)
    color_data[n] = color_image
    color_ts.append(color_time)
    # メモリに保存されたデータをSD カードに書き込むための関数
def Recording():
    for i in range(frame):
        np.save(col_dir + str(i) + ".npy", color_data[i])
        np.save(dep_dir + str(i) + ".npy", depth_data[i])
        np.save(col_dir + "ts.npy", color_ts)
        np.save(dep_dir + "ts.npy", depth_ts)

# ループ処理
frame = 0
try:
    while (frame < frame_limit):
        # 画像取得・メモリに保存
        Measure(frame)
        frame = frame +1
　　
        # SD カードに保存
        Recording()
finally:
    pipeline.stop()
