# 2D-Irregular-Packing-Algorithm

**This repository contains algorithms for 2D irregular packing and a simple tutorial to the algorithms. In our last experiments, most of them work well now. We will update the algorithms then that are still unfinished. Welcome to fork and be one of our contributors.**

## Introduction

Literature Review：[排样问题文献综述/教程（中文）](https://seanys.github.io/2020/03/17/排样问题综述/)  Yang shan 

Author: [Shan Yang](https://github.com/seanys), [Zilu Wang](https://github.com/Prinway) (Department of Science and Managment, Tongji University)

Email: tjyangshan@gmail.com, prinway1226@gmail.com

Yang is working in [Dr. Xiaolei Wang](https://scholar.google.com/citations?user=EN1lJkoAAAAJ&hl=zh-CN)'s laboratory on optimization problems in urban transportation. Wang is working in [Dr. Zhaolin Hu](scholar.google.com/citations?user=AOlc1mMAAAAJ&hl=en)'s laboratory on random optimization and simulation. Our work in 2D irregular packing is encouraged by [Dr. Zhe Liang](https://sem.tongji.edu.cn/semch/15381.html) in our department. Welcome to chat with us on these topics. 

## Dataset

EURO Dataset：https://www.euro-online.org/websites/esicup/data-sets/#1535972088237-bbcb74e3-b507

Data sets are processed to csv in folder [data](data). Use pandas and json loads. 

```python
import pandas as pd
import json
df = pd.read_csv("data/blaz1.csv")
all_polys = []
for i in range(df.shape[0]):
  all_polys.append(df['polygon'][i])
print(all_polys)
```

## Alglorithm

heuristic.py: Bottom-Left-Fill、TOPOS（Debug）

sequence.py: genetic algorithm, simulated annealing

nfp_test.py: try no-fit polygon

lp_algorithm.py: compaction and separation algorithm

lp_search.py: A new

C++ Nesting Problem: C++ Version

## 实现情况 

### 基础算法

- [x] **No-fit Polygon**：基本实现，Start Point的部分暂未实现，参考论文 Burke E K , Hellier R S R , Kendall G , et al. Complete and Robust No-Fit Polygon Generation for the Irregular Stock Cutting Problem[J]. European Journal of Operational Research, 2007, 179(1):27-49.

形状A固定位置，在B上选择一个参考点比如左下角点P，形状B紧贴着绕A一周，P的轨迹会形成一个形状即NFP，P在NFP上或外部，则该排列可行；如果P点在NFP内，则方案不可行（[图片来源](https://github.com/Jack000/SVGnest)）

<img src="https://camo.githubusercontent.com/1156f6f8323c52dea2981604dd780b02add19e86/687474703a2f2f7376676e6573742e636f6d2f6769746875622f6e66702e706e67" alt="img" width="500px" />



### 序列排样

- [x] Bottom Left Fill：已经实现，参考论文 

a. 选择一个形状加入，通过计算inner fit polygon，也就是形状绕着Bin/Region内部一周，参考点P会形成的一个长方形，P点在该长方形内部则解是feasible solution 

b. 选择能够摆到的最左侧位置，放进去即可 （[图片来源](https://github.com/Jack000/SVGnest)）

<img src="https://camo.githubusercontent.com/f7973d894432676e37c3489c3248c3a31cf3e945/687474703a2f2f7376676e6573742e636f6d2f6769746875622f6e6670322e706e67" alt="No Fit Polygon example" width="500px"/>

- [x] TOPOS：已经实现，参考论文：
- [x] GA/SA：两个优化算法优化顺序已经实现

### 基于布局的优化

- [x] Fast Neighborhood Search：基本实现，有一些Bug还需要修改
- [x] Cuckoo Search：基本算法已经实现，需要进一步完善
- [x] ~~ILSQN：不准备写了，没太大意义~~
- [x] ~~Guided Local Search：同上~~

### 线性规划排样

- [x] Compaction：压缩边界，已经实现
- [x] Separation：去除重叠，已经实现
- [ ] SHAH：基于模拟退火算法和上述两个算法的Hybrid Algorithm，暂时未做

## Reference Paper

Topos 