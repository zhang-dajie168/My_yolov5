#!/bin/bash
set PYTHONIOENCODING=utf-8

# 删除当前文件夹下的VOC多级文件
rm -r VOCdevkit



echo "***********开始数据增强*********"
echo "############"
cd  'E:\yolov5\DataAugForObjectDetection'
conda activate yolov5
DataAugmentforLabelImg.py
echo "############"
echo "***********数据增强完成*********"

sleep 3

echo "********************"
echo "********************"
echo "********************"
echo "***********准备划分数据集*********"
cd ..
conda activate yolov5
prepare_data.py
echo "***********数据集划分完成*********"


sleep 3

echo "********************"
echo "********************"
echo "********************"
echo "***********开始训练模型*********"
conda activate yolov5
python train.py --epochs 200

echo "***********模型训练完成*********"
