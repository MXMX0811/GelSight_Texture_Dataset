import sys
import numpy as np
import cv2
import os
import re
import pickle
from gsrobotics.examples import gsdevice, gs3drecon


# def find_window(frame, threshold, tolerance = 80):
#     """
#     根据RGB阈值动态检测窗口范围。
#     :param frame: 输入的彩色帧
#     :param threshold: 黑色区域的RGB阈值 (R, G, B)
#     :return: 窗口的 x, y, w, h
#     """
#     threshold = np.array(threshold)
#     diff = np.linalg.norm(frame - threshold, axis=-1)

#     # 创建掩码：色差小于容差的区域
#     mask = diff <= tolerance

#     # 创建一个掩码，找到大于阈值的区域（非黑色区域）
#     # mask = np.all((frame < threshold + 40) & (frame > threshold - 40), axis=-1)
#     coords = np.column_stack(np.where(mask))  # 获取非黑色区域坐标
    
#     # print(coords.size)

#     if coords.size == 0:
#         # 如果没有找到非黑色区域，返回整个图像
#         return 0, 0, frame.shape[1], frame.shape[0]
    
#     # 获取窗口范围
#     x, y, w, h = (
#         coords[:, 1].min(),
#         coords[:, 0].min(),
#         coords[:, 1].max() - coords[:, 1].min(),
#         coords[:, 0].max() - coords[:, 0].min(),
#     )
    
#     return x, y, w, h

import cv2
import numpy as np

def find_window(frame):
    """
    使用轮廓树检测窗口范围（替代原先颜色阈值法）
    :param frame: 输入彩色帧
    :return: 窗口的 (x, y, w, h)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape[:2]

    # 1) 预处理
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    bin_ = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 5
    )

    # 2) 闭运算连通边界
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    bin_ = cv2.morphologyEx(bin_, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 3) 查找轮廓树
    cnts, hier = cv2.findContours(bin_, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None or len(cnts) == 0:
        return 0, 0, W, H
    hier = hier[0]

    def approx_quad(c):
        peri = cv2.arcLength(c, True)
        return cv2.approxPolyDP(c, 0.02 * peri, True)

    outer_idx = -1
    inner_idx = -1

    # 4) 寻找外框（面积较大且不贴边的四边形）
    for i, (c, h) in enumerate(zip(cnts, hier)):
        a = cv2.contourArea(c)
        if a < 0.2 * H * W or a > 0.95 * H * W:
            continue
        ap = approx_quad(c)
        if len(ap) == 4:
            x, y, w, hb = cv2.boundingRect(ap)
            margin = min(x, y, W - (x + w), H - (y + hb))
            if margin > 0.02 * max(W, H):
                outer_idx = i
                break

    # 5) 在外框的子轮廓中寻找内框
    if outer_idx != -1:
        child = hier[outer_idx][2]  # 第一个子轮廓
        best = (-1, -1)
        while child != -1:
            c = cnts[child]
            ap = approx_quad(c)
            if len(ap) == 4:
                a_in = cv2.contourArea(ap)
                a_out = cv2.contourArea(cnts[outer_idx])
                ratio = a_in / a_out
                if 0.2 < ratio < 0.95:
                    x, y, w, hb = cv2.boundingRect(ap)
                    rect_area = w * hb
                    rectangularity = a_in / (rect_area + 1e-6)
                    score = rectangularity
                    if score > best[0]:
                        best = (score, child)
            child = hier[child][0]
        inner_idx = best[1]

    # 6) 返回内框的矩形范围
    if inner_idx == -1:
        # 如果没找到，退化为整图
        return 0, 0, W, H

    x, y, w, h = cv2.boundingRect(cnts[inner_idx])
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
    
    rgb_cap = cv2.VideoCapture(0)
    if not rgb_cap.isOpened():
        print("Cannot open rgb camera")
        exit()

    print('roi = ', roi)
    print('press q on image to exit')

    ''' use this to plot just the 3d '''
    vis3d = gs3drecon.Visualize3D(dev.imgh, dev.imgw, '', mmpp)
    
    # texture_path = input('Input the texture label: ')
    texture_path = 'Mat'
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
            
            H, W, _ = rgb_frame.shape
            # 图像中心坐标
            center_y = H // 2
            center_x = W // 2

            # 获取中心 4x4 区域的起止坐标
            start_y = center_y - 3
            end_y = center_y + 3
            start_x = center_x - 3
            end_x = center_x + 3

            # 提取中心 4x4 区域的像素块
            center_patch = rgb_frame[start_y:end_y, start_x:end_x]  # 形状为 (6, 6, 3)

            # 计算平均 RGB 值
            # mean_rgb = center_patch.mean(axis=(0, 1))  # 形状为 (3,)
            # crop_x, crop_y, crop_w, crop_h = find_window(rgb_frame, mean_rgb, 10)
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
            if f1 is None:
                print("No gel frame!")
                continue
            # bigframe = cv2.resize(f1, (f1.shape[1] * 2, f1.shape[0] * 2))
            # cv2.imshow('Image', bigframe)

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
