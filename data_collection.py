import sys
import numpy as np
import cv2
import os
import re
import pickle
from gsrobotics.examples import gsdevice, gs3drecon


def find_window(frame, threshold=(80, 80, 80)):
    """
    根据RGB阈值动态检测窗口范围。
    :param frame: 输入的彩色帧
    :param threshold: 黑色区域的RGB阈值 (R, G, B)
    :return: 窗口的 x, y, w, h
    """
    # 创建一个掩码，找到大于阈值的区域（非黑色区域）
    mask = np.all(frame < threshold, axis=-1)
    coords = np.column_stack(np.where(mask))  # 获取非黑色区域坐标
    
    # print(coords.size)

    if coords.size == 0:
        # 如果没有找到非黑色区域，返回整个图像
        return 0, 0, frame.shape[1], frame.shape[0]
    
    # 获取窗口范围
    x, y, w, h = (
        coords[:, 1].min(),
        coords[:, 0].min(),
        coords[:, 1].max() - coords[:, 1].min(),
        coords[:, 0].max() - coords[:, 0].min(),
    )
    
    return x, y, w, h

def main(argv):
    # Set flags
    SAVE_VIDEO_FLAG = False
    FIND_ROI = False
    GPU = False
    MASK_MARKERS_FLAG = False
    
    # def list_available_cameras(max_index=10):
    #     available_cameras = []
    #     for i in range(max_index):
    #         cap = cv2.VideoCapture(i)
    #         if cap.isOpened():
    #             available_cameras.append(i)
    #             cap.release()
    #     return available_cameras

    # cameras = list_available_cameras()
    # print("Available cameras:", cameras)

    # Path to 3d model
    path = '.'

    # Set the camera resolution
    mmpp = 0.0634  # mini gel 18x24mm at 240x320

    # the device ID can change after unplugging and changing the usb ports.
    # on linux run, v4l2-ctl --list-devices, in the terminal to get the device ID for camera
    dev = gsdevice.Camera("GelSight Mini")
    net_file_path = 'gsrobotics/examples/nnmini.pt'

    dev.connect()

    ''' Load neural network '''
    model_file_path = path
    net_path = os.path.join(model_file_path, net_file_path)
    print('net path = ', net_path)

    if GPU:
        gpuorcpu = "cuda"
    else:
        gpuorcpu = "cpu"

    nn = gs3drecon.Reconstruction3D(dev)
    net = nn.load_nn(net_path, gpuorcpu)

    f0 = dev.get_raw_image()
    roi = (0, 0, f0.shape[1], f0.shape[0])
    
    rgb_cap = cv2.VideoCapture(1)
    if not rgb_cap.isOpened():
        print("Cannot open rgb camera")
        exit()

    print('roi = ', roi)
    print('press q on image to exit')

    ''' use this to plot just the 3d '''
    vis3d = gs3drecon.Visualize3D(dev.imgh, dev.imgw, '', mmpp)
    
    # texture_path = input('Input the texture label: ')
    texture_path = 'Leather'
    texture_path = 'Texture/' + texture_path
    if not os.path.exists(texture_path):
        os.makedirs(texture_path)
        print(f"Make the dir: {path}")
    
    max_number = 0
        
    for filename in os.listdir(texture_path):
        numbers = re.findall(r'\d+', filename)
        if numbers:
            numbers = [int(num) for num in numbers]
            max_file_number = max(numbers)
            if max_number is None or max_file_number > max_number:
                max_number = max_file_number
    
    sample_index = max_number + 1
    save_flag = 0
    
    print('Press enter to save rgb, press enter again to save height map')

    try:
        while dev.while_condition:
            ret, rgb_frame = rgb_cap.read()
            if not ret:
                print("No rgb frame")
                break
            
            crop_x, crop_y, crop_w, crop_h = find_window(rgb_frame)
            if crop_w == 0 or crop_h  == 0:
                continue
            crop_w = int(crop_h / 3 * 4)
            cropped_frame = rgb_frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]

            # 调整分辨率到 320x240
            resized_frame = cv2.resize(cropped_frame, (320, 240))
            
            # 显示实时画面
            cv2.imshow('Cropped RGB', resized_frame)

            # get the roi image
            f1 = dev.get_image()
            bigframe = cv2.resize(f1, (f1.shape[1] * 2, f1.shape[0] * 2))
            #cv2.imshow('Image', bigframe)

            # compute the depth map
            dm = nn.get_depthmap(f1, MASK_MARKERS_FLAG)
            # print(dm.max())
            cv2.imshow('Image', dm/dm.max())

            ''' Display the results '''
            vis3d.update(dm)
            
            key = cv2.waitKey(100)
            if key == 13 and save_flag == 0:  # Enter 1, save rgb
                image_name = texture_path + '/' + str(sample_index) + '.jpg'
                # select gelsight working area
                cropped_rate = 0.3
                cropped_frame_gs = cropped_frame[int(crop_h*(cropped_rate/2)):int(crop_h*(1-cropped_rate/2)), 
                                                 int(crop_w*(cropped_rate/2)):int(crop_w*(1-cropped_rate/2))]
                resized_frame_gs = cv2.resize(cropped_frame_gs, (320, 240))
                cv2.imwrite(image_name, resized_frame_gs)
                print(f"Saved rgb {image_name}")
                print("Press enter again to save height map")
                save_flag = 1
                key = -1
            if key == 13 and save_flag == 1:  # Enter 2, save height map
                image_name = texture_path + '/' + str(sample_index) + '.pkl'
                with open(image_name, "wb") as f:  # 使用 "wb" 表示写入二进制文件
                    pickle.dump(dm, f)
                print(f"Saved height map {image_name}")
                print("Press enter to save next sample")
                save_flag = 0
                sample_index += 1
                key = -1

    except KeyboardInterrupt:
        print('Interrupted!')
        dev.stop_video()


if __name__ == "__main__":
    main(sys.argv[1:])
