# 目标检测网络 CenterNet

# 需求解读
        随着基于Anchor的目标检测性能达到了极限，基于Anchor-free的目标检测算法成为了当前的研究热点，具有代表性的工作包括CornerNet、FOCS与CenterNet等。
        除此之外，基于Anchor的目标检测算法存在着一些严重的问题，具体包括：
        1、Anchros的定义在一定程度上会限制检测算法的性能；
        2、NMS等后处理操作会降低整个检测算法的速度。
        为了解决这些问题，基于Anchor-free的目标检测算法应运而生。

## anchor free
    什么是anchor？
        yolo ssd fast-rcnn有anchor的概念，把feature map分成一些网格，在网格上画出一些框来，每一个网格上面再画出一个框来，预测时候用最接近物体的框的预测它。
        绿色框是(anchor box)预测框，黄色框是ground truth。网络输出会把绿色框左移，高度调整 -- 带anchor网络的预测过程。
![anchor box](./anchor%20box.png)

# CenterNet算法简介
        CenterNet是一个基于Anchor-free的目标检测算法，该算法是在CornerNet算法的基础上改进而来的。与单阶段目标检测算法yolov3相比，该算法在保证速度的前提下，精度提升了4个百分点。与其它的单阶段或者双阶段目标检测算法相比，该算法具有以下的优势：
        1、该算法去除低效复杂的Anchors操作，进一步提升了检测算法性能；
        2、该算法直接在heatmap图上面执行了过滤操作，去除了耗时的NMS后处理操作，进一步提升了整个算法的运行速度；
        3、该算法不仅可以应用到2D目标检测中，经过简单的改变它还可以应用3D目标检测与人体关键点检测等其它的任务中，即具有很好的通用性。

## CenterNet原理
    图像传入全卷积网路，得到一个热力图，热力图峰值点即中心点，每个特征图的峰值点地址预测了目标的宽高信息。
![CenterNet原理](./CenterNet%E5%8E%9F%E7%90%86.png)
CenterNet网络机构
![CenterNet网络结构](./CenterNet%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84.png)

### headmap(热力图)的生成
    两个猫的中心点就是热力图的中心点，每一个圆都是一个高斯圆，由中心点向外减弱，规则是高斯函数。
![热力图的生成](./%E7%83%AD%E5%8A%9B%E5%9B%BE%E7%9A%84%E7%94%9F%E6%88%90.png)






# 参考
    https://blog.csdn.net/WZZ18191171661/article/details/113753991


