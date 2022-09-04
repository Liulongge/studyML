# 目标检测网络 CenterNet

## 特点：anchor free
    什么是anchor？
        yolo ssd fast-rcnn有anchor的概念，把feature map分成一些网格，在网格上画出一些框来，每一个网格上面再画出一个框来，预测时候用最接近物体的框的预测它。
        绿色框是(anchor box)预测框，黄色框是ground truth。网络输出会把绿色框左移，高度调整 -- 带anchor网络的预测过程。
![anchor box](./anchor%20box.png)

## CenterNet原理
    图像传入全卷积网路，得到一个热力图，热力图峰值点即中心点，每个特征图的峰值点地址预测了目标的宽高信息。
![CenterNet原理](./CenterNet%E5%8E%9F%E7%90%86.png)
CenterNet网络机构
![CenterNet网络结构](./CenterNet%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84.png)

### headmap(热力图)的生成
    两个猫的中心点就是热力图的中心点，每一个圆都是一个高斯圆，由中心点向外减弱，规则是高斯函数。
![热力图的生成](./%E7%83%AD%E5%8A%9B%E5%9B%BE%E7%9A%84%E7%94%9F%E6%88%90.png)







