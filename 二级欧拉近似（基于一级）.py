# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 21:15:04 2023

@author: 86319
"""

# 引入 numpy 和 matplotlib.pyplot 库
import numpy as np
import matplotlib.pyplot as plt

# 定义微分方程 
def f(y, t):
    return 1+np.exp(-t)*np.sin(y)

# 定义二级欧拉近似方法
def euler_2(f, t0, y0, h, num):
   
    # 创建 x 和 y 的数组,x1为位移数组
    y = np.zeros(num)
    t = np.zeros(num)
    
    # 将初始值放入数组
    y[0] = y0
    t[0] = 0   
    # 求解微分方程
    for i in range(1, num):
        y_half = y[i - 1] + h * f(y[i - 1],t[i - 1])
        y[i] = y[i - 1] + h / 2 * (f( y[i - 1],t[i - 1] + h) + f(y_half,t[i - 1] + h))
        t[i] = t[i - 1] + h
    # 返回结果
    return y, t

# 定义初始条件和计算参数
x0 = 4
y0 = 0
h = 0.1
num = 10

# 使用二级欧拉近似方法求解微分方程
y ,t = euler_2(f, x0, y0, h, num)

# 输出数值解
print("数值解：", y)

# 绘制摆角与时间关系图像
plt.plot(t, y ,'-o')
plt.title("y' =1+exp(-t)*sin(y), y(0) = 0")  # 绘制图像的标题
plt.xlabel('t')   # 绘制 x 轴的标签
plt.ylabel('y')   # 绘制 y 轴的标签
plt.show()        # 显示图像