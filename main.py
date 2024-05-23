# Importing necessary libraries
import cv2
import numpy as np 
import pickle

# 加载停车位坐标
with open('car_park_pos', 'rb') as f:
        pos_list = pickle.load(f)

# 每个停车位的大小
width, height = 27, 15


#  Function 检查给定帧中每个停车位的状态
def check_parking_space(img):
    free_spaces = 0

    # 循环每个停车位坐标
    for pos in pos_list:
        # 裁剪图像以获得停车位区域
        img_crop = img[pos[1]:pos[1] + height, pos[0]:pos[0] + width]                       
        count = cv2.countNonZero(img_crop)

        if count > 110:
            color = (0, 0, 255)
       
        else:
            free_spaces += 1
            color = (0, 255, 0)

        # 在停车位周围绘制一个矩形，并显示非零像素的计数
        cv2.rectangle(frame, pos, (pos[0] + width, pos[1] + height), color, 1)
        cv2.putText(frame, str(count), (pos[0], pos[1] + height - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, color, 1)

    # 显示停车位总数中的可用停车位总数
    cv2.putText(frame, f'{free_spaces} / {len(pos_list)}', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,255), 3)


cap = cv2.VideoCapture("busy_parking_lot.mp4")

# 获取视频帧的尺寸
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30

# 设置视频写入程序，将处理后的视频写入文件
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # mp4 codec
out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

while 1:
        #  从视频捕获中读取帧
        ret, frame = cap.read()

        # 将帧转换为灰度
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     
        # 使用高斯滤波器模糊灰度帧
        blurred_frame = cv2.GaussianBlur(gray_frame, (3, 3), 1)

        # 应用自适应阈值对模糊帧进行二值化#,
        threshold_frame = cv2.adaptiveThreshold(blurred_frame, 255,
                                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY_INV, 25, 16)

        # 对阈值帧应用中值滤波去除噪声
        frame_median = cv2.medianBlur(threshold_frame, 5)

        # Dilating the filtered frame to fill in gaps in the parking space boundaries
        # 展开过滤后的框架以填充停车位边界中的间隙
        kernel = np.ones((5, 5), np.uint8)
        dilated_frame = cv2.dilate(frame_median, kernel, iterations=1)
        
        check_parking_space(dilated_frame)

        cv2.imshow('frame', frame)

        if cv2.waitKey(20) & 0xFF == ord("q"):
               break

cap.release()
out.release()
cv2.destroyAllWindows()