import time
import cv2
import numpy as np
import math


class Xianlan_Process():
    def __init__(self):
        self.distance = 100  # 相邻两个轮廓中心点距离的阈值
        self.ok_number = 6
        self.green = (0, 255, 0)
        self.red = (0, 0, 255)

    def contours_rect(self, contours):
        rect_contours = []
        for i, contour in enumerate(contours):
            # cls = contour[0]
            segment = contour[1:].reshape(int(len(contour) / 2), 1, 2)  # 转为opencv 轮廓类型
            rect = cv2.minAreaRect(segment)  # 获取最小外接矩形
            rect_contours.append(rect)
        rect_contours.sort(key=lambda x: x[0][1])  # 按照中心点y从小到大排序
        # print(rect_contours)
        return rect_contours

    def draw_rect_box(self, img, rect, color=(0, 255, 0)):
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        cv2.drawContours(img, [box], 0, color, 2)  # 画出最小矩形轮廓

    def distance_point(self, point1, point2):
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def count_number_ok(self, contours, image):
        flage = True
        if len(contours) == 0:
            return image, flage
        rect_contours = self.contours_rect(contours)
        if len(rect_contours) > self.ok_number:
            flage = False
            for i in range(len(rect_contours)):
                self.draw_rect_box(image, rect_contours[i], color=self.red)
            cv2.putText(image, "NG", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            for i in range(len(rect_contours)):
                self.draw_rect_box(image, rect_contours[i], color=self.green)
            cv2.putText(image, "OK", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return image, flage

    def main(self, contours, image):
        flage = []
        if len(contours) <= 1:
            return image, False not in flage
        rect_contours = self.contours_rect(contours)
        for i in range(len(rect_contours) - 1):
            rect1 = rect_contours[i]
            rect2 = rect_contours[i + 1]
            distance = self.distance_point(rect1[0], rect2[0])
            print(f"distance:{distance}")
            if distance < self.distance:
                flage.append(False)
                self.draw_rect_box(image, rect1, color=self.red)
            # else:
            # self.draw_rect_box(image, rect1, color=self.green)
        print(flage)
        return image, False not in flage


if __name__ == '__main__':
    from api3 import Yolov5_Seg

    yolov5 = Yolov5_Seg(save_path='/home/ymt/data/26.线圈缺陷检测/xianquan-seg-0729.pt',
                        device='cuda:0',
                        confidence_threshold=0.8)
    dt = Xianlan_Process()
    st = time.time()
    image = cv2.imread('/home/ymt/data/26.线圈缺陷检测/test_img/xianquan_1_250.jpg',0)
    # image=cv2.cvtColor()
    print(image.shape)
    rec = yolov5.cn(image)
    print(time.time() - st)
    contours, boxs = yolov5.predict(rec, image)
    print(boxs)
    # print(contours)

    # image, flage = dt.main(contours, image)
    image, flage = dt.count_number_ok(contours, image)
    print(flage)
    print("time", time.time() - st)
    # cv2.imwrite("output3.jpg", image)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # # **********视频处理***********
    # 读取视频文件
    # video_capture = cv2.VideoCapture('/home/ymt/data/26.线圈缺陷检测/vieo-0726/Video_20240726162001118.avi')
    #
    # # 检查视频是否成功打开
    # if not video_capture.isOpened():
    #     print("Error: Unable to open video file.")
    #     exit()
    #
    # # 获取视频的基本信息
    # fps = video_capture.get(cv2.CAP_PROP_FPS)
    # frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    #
    # # 创建用于保存视频的VideoWriter对象
    # output_video = cv2.VideoWriter('output_video3.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps,
    #                                (frame_width, frame_height))
    #
    # # 读取视频帧并处理
    # while True:
    #     ret, frame = video_capture.read()
    #     if not ret:
    #         break
    #     rec = yolov5.cn(frame)
    #     contours, boxs = yolov5.predict(rec, frame)
    #     # image, flage = dt.main(contours, frame)
    #     image, flage = dt.count_number_ok(contours, frame)
    #
    #     # 将处理后的帧写入输出视频文件
    #     output_video.write(image)
    #
    #     # 可选：显示处理后的视频帧
    #     cv2.imshow('Processed Video', image)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    #
    # # 释放资源
    # video_capture.release()
    # output_video.release()
    # cv2.destroyAllWindows()
