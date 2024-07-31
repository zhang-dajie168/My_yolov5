import time
import cv2
import numpy as np
import math


class Contours_Process():
    def __init__(self):
        self.distance = 100
        self.vertical_up = [0, 100]  # 竖直向上中心向量
        self.area_dt = True  # 是否面积过滤
        self.area_roi = [4000, 10000]  # 轮廓面积的区间

    def drawLine(self, image, start_point, angle_red, color=(0, 0, 255), length=100):
        # start_point = center
        # 定义向量长度和角度
        # length = 100
        # angle_degrees = -45  # 顺时针旋转45度(弧度)
        angle_radians = -angle_red
        # 计算终点坐标
        end_point = (
            int(start_point[0] + length * math.cos(angle_radians)),
            int(start_point[1] - length * math.sin(angle_radians)))
        # 画出向量
        cv2.arrowedLine(image, start_point, end_point, color, 2, tipLength=0.1)

    def area_colation(self, contours, det_array):
        contours_area = []
        area_array = []
        for i, contour in enumerate(contours):
            # cls = contour[0]
            segment = contour[1:].reshape(int(len(contour) / 2), 1, 2)  # 转为opencv 轮廓类型
            area = cv2.contourArea(segment)
            print(f"area:{area}")
            if area >= self.area_roi[0] and area <= self.area_roi[1]:
                contours_area.append(contour)
                area_array.append(det_array[i])
        return contours_area, area_array

    def contours_rect(self, contours, image, det_array):
        mango_rect = []
        apple_array = []
        # 遍历每个轮廓
        if self.area_dt:
            contours, det_array = self.area_colation(contours, det_array)
        for i, contour in enumerate(contours):
            cls = contour[0]
            conf = det_array[i][-2]
            if cls == 1:
                segment = contour[1:].reshape(int(len(contour) / 2), 1, 2)  # 转为opencv 轮廓类型
                rect = cv2.minAreaRect(segment)  # 获取最小外接矩形，并画图
                box = cv2.boxPoints(rect)
                box = np.intp(box)
                cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
                center = (int(rect[0][0]), int(rect[0][1]))
                width = rect[1][0]
                height = rect[1][1]

                # 计算角度
                center1, center2 = self.get_center_point(contour)
                distances_and_points = self.sort_point(center1, center2,
                                                       (image.shape[0] // 2, image.shape[1]))
                cv2.arrowedLine(image, distances_and_points[0][1].astype(int),
                                distances_and_points[1][1].astype(int),
                                color=(255, 0, 0), thickness=2, tipLength=0.1)
                point1, point2 = distances_and_points[0][1].astype(int), distances_and_points[1][1].astype(int)
                vector = [point2[0] - point1[0], point2[1] - point1[1]]  # 向量，靠近底边中心的点，指向向上
                # print("point1", point1, "point2", point2, "向量", vector)
                angel_deg = self.direction_with_vertical_up(vector)  # 计算与水平向上角度，左负右正
                # print(f"角度:{angel_deg}")
                cv2.putText(image, f"{angel_deg:.1f}", center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                mango_rect.append((center, (width, height), angel_deg, conf))
            else:
                x1, y1, x2, y2 = det_array[i][:4]
                # print(x1, y1, x2, y2)
                image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)  # 绘制矩形
                apple_array.append(det_array[i])
        return image, mango_rect, apple_array

    def distance_point(self, point1, point2):
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def sort_point(self, point1, point2, center_point):
        # 计算每个点到中心点的距离
        distance_to_center1 = self.distance_point(center_point, point1)
        distance_to_center2 = self.distance_point(center_point, point2)

        # 构建距离和点的元组列表
        distances_and_points = [
            (distance_to_center1, point1),
            (distance_to_center2, point2)
        ]
        # 按照距离从小到大排序
        distances_and_points.sort(key=lambda x: x[0])
        return distances_and_points

    def get_center_point(self, contour):
        segment = contour[1:].reshape(int(len(contour) / 2), 1, 2)  # 转为opencv 轮廓类型
        # 计算最小外接矩形
        rect = cv2.minAreaRect(segment)
        width = rect[1][0]
        height = rect[1][1]
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        # 获取短边中心点
        if width > height:
            center1 = (box[0] + box[1]) / 2
            center2 = (box[2] + box[3]) / 2
        else:
            center1 = (box[1] + box[2]) / 2
            center2 = (box[0] + box[3]) / 2
        return center1, center2

    def distances_and_points(self, contours, img):
        center_points = []
        if len(contours) == 0:
            return img, center_points

        for contour in contours:
            # cls = contour[0]
            segment = contour[1:].reshape(int(len(contour) / 2), 1, 2)  # 转为opencv 轮廓类型
            # 计算最小外接矩形
            rect = cv2.minAreaRect(segment)
            # width = rect[1][0]
            # height = rect[1][1]
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # print("box", box, "w", width, "h", height)
            cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
            center1, center2 = self.get_center_point(contour)

            distances_and_points = self.sort_point(center1, center2,
                                                   (img.shape[0] // 2, img.shape[1]))
            cv2.arrowedLine(img, distances_and_points[0][1].astype(int), distances_and_points[1][1].astype(int),
                            color=(255, 0, 0), thickness=2, tipLength=0.1)
            center_points.append(distances_and_points)
        return img, center_points

    def select_contours(self, center_contours):
        center_contours = sorted(center_contours, key=lambda x: x[0][0])  # 先按照距离进行升序排序
        contours = None
        if len(center_contours) >= 2 and self.distance_point(center_contours[0][0][1],
                                                             center_contours[0][1][1]) <= self.distance:
            contours = center_contours[1]
        else:
            contours = center_contours[0]
        return contours

    def angle_with_vertical_up(self, vector):
        # 计算向量的模长
        magnitude = math.sqrt(vector[0] ** 2 + vector[1] ** 2)
        magnitude_vector = math.sqrt(self.vertical_up[0] ** 2 + self.vertical_up[1] ** 2)

        dot_product = np.dot(vector, self.vertical_up)  # 计算向量与竖直向上中心向量的点积
        angle_radians = np.arccos(np.clip(dot_product / (magnitude * magnitude_vector), -1.0, 1.0))  # 计算夹角（弧度）
        angle_degrees = math.degrees(angle_radians)  # 转换弧度为角度
        return angle_degrees

    def direction_with_vertical_up(self, vector):
        # 获取向量相对竖直向上向量的偏移角度
        angle = self.angle_with_vertical_up(vector)
        # 确定偏移方向
        if vector[0] >= 0:
            angle = 180 - angle
        else:
            angle = angle - 180
        return angle

    def main_dt_angel(self, contours, image):  # 车道线检测角度
        if len(contours) == 0:
            return image, None, None
        img, center_points = self.distances_and_points(contours, image)
        contour = self.select_contours(center_points)
        point1, point2 = contour[0][1], contour[1][1]
        vector = [point2[0] - point1[0], point2[1] - point1[1]]
        print("point1", point1, "point2", point2, "向量", vector)
        angle = self.direction_with_vertical_up(vector)
        return image, angle, point1


if __name__ == '__main__':
    from api3 import Yolov5_Seg

    yolov5 = Yolov5_Seg(save_path='/home/ymt/ArmPickPlace/yolov5/weights/seg-apple_mango-0708.pt',
                        device='cuda:0',
                        confidence_threshold=0.9)
    dt = Contours_Process()
    st = time.time()
    image = cv2.imread('/home/ymt/yolov5/dataset/images/test/captured_image_18_8.jpg')
    # print(image.shape)
    rec = yolov5.cn(image)
    print(time.time() - st)
    contours, boxs = yolov5.predict(rec, image)
    print(boxs)
    # print(contours)

    image, mango_rect, apple_array = dt.contours_rect(contours, image, boxs)
    print("mango_rect", mango_rect)
    print("apple_array", apple_array)
    # image, angle, point1 = dt.main_dt_angel(contours, image)
    # cv2.putText(image, f"{angle:.1f}", (image.shape[0] // 2, image.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1,
    #             (0, 0, 255), 2)
    print("time", time.time() - st)
    # cv2.imwrite("output3.jpg", image)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # # # **********视频处理***********
    # # 读取视频文件
    # video_capture = cv2.VideoCapture('/home/ymt/视频/line_avi/record_20240626103034.avi')
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
    # output_video = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps,
    #                                (frame_width, frame_height))
    #
    # # 读取视频帧并处理
    # while True:
    #     ret, frame = video_capture.read()
    #     if not ret:
    #         break
    #     rec = yolov5.cn(frame)
    #     contours, boxs = yolov5.predict(rec, frame)
    #     image, angle, point1 = dt.main_dt_angel(contours, frame)
    #     if angle is not None:
    #         cv2.putText(image, f"{angle:.2f}", (int(point1[0]), int(point1[1] - 30)), cv2.FONT_HERSHEY_SIMPLEX, 1,
    #                     (0, 0, 255), 2)
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
