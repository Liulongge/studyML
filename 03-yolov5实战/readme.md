# yolov5
        目标检测(object detection) = what and where
    识别: what(recognition)
    定位: where(localization), 位置(最小外接矩形, bounding box)
    类别标签(category label)
    置信度得分(confidence score)
![目标检测](./doc/目标检测.jpg)

# 定位和检测
    定位: 找到检测图像中带有一个给定标签的单目标
    检测: 找到图像中带有给定标签的所有目标
![定位和检测](./doc/定位和检测.jpg)

# 目标检测常用数据集
## PASCAL VOC
        PASCAL VOC挑战赛在2005年至2012年展开。
        PASCAL VOC 2007: 9963张图像，24640个标注；PASCAL VOC 2012：11530张图像，27450个标注，该数据集有20个分类：
        Person: person
        Animal: bird, cat, cow, dog, horse, sheep
        Vehicle(交通工具): aeroplane, bicycle, boat, bus, car, motorbike, train
        Indoor(室内物体): bottle, chair, dining table, potted plant, sofa, tv/mointor
        链接: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

## MS COCO
        MS COCO全称是Microsoft Common Objects in Context，起源是微软于2014年出资标注的Microsoft COCO数据集，与ImageNet竞赛一样，被视为是计算机视觉领域最受关注和最权威的比赛之一。
        在ImageNet竞赛停办后，COCO竞赛就成为当前目标识别、检测等领域的一个最权威、最重要的标杆，也是目前该领域在国际上唯一能汇聚Google、微软、Facebook以及国内外众多顶尖院校和优秀创新企业共同参与的大赛。
        COCO数据集包含20万个图像：11.5万多张训练集图像，5千张验证集图像，2万张测试集图像
        80个类别中有超50万个目标标注，平均每个图像的目标数为7.2个。
        链接: https://cocodataset.org/#home

## 目标检测性能指标
### 检测精度
        Precision(精度/查准率，准不准), recall(召回率/查全率，全不全), F1 score
        Iou(Intersection over Union)
        P-R curve(Precision-Recall curve)
        AP(Average Precision)
        mAP(mean Average Precision)
        混淆矩阵：
![混淆矩阵](./doc/混淆矩阵.jpg)
![混淆矩阵2](./doc/混淆矩阵2.jpg)
### 检测速度
        前传耗时
        每秒帧数FPS(Frame Per Second)
        浮点运算量(FLOPS)


