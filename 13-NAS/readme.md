# NAS: network architecture search
        网络架构搜索，三大基本要件：
        1. 在什么范围内搜索(搜索空间search space)
        2. 如何确定搜索路径(搜索策略search strategy)
        3. 怎么确定结果好坏(评估策略evaluation strategy)
## NAS解决的数学问题
        神经网络就是一个x变成y的函数，从而变成一个有向无环图。根本上，这些计算单元决定了网络的结构，而NAS就是要找到最优的计算单元以及他们的组合。
![NAS解决的数学问题](./docs/NAS解决的数学问题.jpg)
## 搜索空间(search space)
### 全局搜索空间
### 基于细胞的搜索空间 
