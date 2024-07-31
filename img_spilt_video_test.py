import os
import cv2


def save_img2(video_path, img_name):  # 提取视频中图片 按照每秒提取   间隔是视频帧率
    video_path = video_path  # 视频所在的路径
    f_save_path = f'{video_path}'  # 保存图片的目录
    videos = os.listdir(video_path)  # 返回指定路径下的文件和文件夹列表。
    for i, video_name in enumerate(videos):  # 依次读取视频文件
        file_name = f'{img_name}_img'  # 拆分视频文件名称 ，剔除后缀
        folder_name = f_save_path + file_name  # 保存图片的上级目录+对应每条视频名称 构成新的目录存放每个视频的
        os.makedirs(folder_name, exist_ok=True)  # 创建存放视频的对应目录
        vc = cv2.VideoCapture(video_path + video_name)  # 读入视频文件
        fps = vc.get(cv2.CAP_PROP_FPS)  # 获取帧率
        print(fps)  # 帧率可能不是整数  需要取整
        rval = vc.isOpened()  # 判断视频是否打开  返回True或False
        c = 1
        count = 1
        while rval:  # 循环读取视频帧
            rval, frame = vc.read()  # videoCapture.read() 函数，第一个返回值为是否成功获取视频帧，第二个返回值为返回的视频帧：
            pic_path = folder_name
            if rval:
                if c % round(fps / 20) == 0:  # 每隔fps帧进行存储操作   ,可自行指定间隔
                    cv2.imwrite(f'{pic_path}/{img_name}_{i}_{count}.jpg', frame)  # 存储为图像的命名 video_数字（第几个文件）.png
                    print(f'{img_name}_{i}_{count}.jpg')
                    count += 1
                cv2.waitKey(1)  # waitKey()--这个函数是在一个给定的时间内(单位ms)等待用户按键触发;如果用户没有按下键,则接续等待(循环)
                c = c + 1
            else:
                break
        vc.release()
        print('save_success' + folder_name)


if __name__ == '__main__':
    img_name = "CL-img"
    video_path = "/home/ymt/ZW_CL/CL_video/"
    save_img2(video_path, img_name)
