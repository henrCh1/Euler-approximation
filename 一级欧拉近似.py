# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 20:55:49 2023

@author: 86319
"""

# 引入 numpy 和 matplotlib.pyplot 库
import numpy as np
import matplotlib.pyplot as plt

# 定义微分方程 dy/dx = f(x, y)
def f(x, y):
    return -0.02*y/0.05-9.81*np.sin(x)/1.016

# 定义一级欧拉近似方法
def euler(f, x0, y0, h, num):
    """
    参数：
    f -- 微分方程 dy/dx = f(x, y)
    x0 -- 初始 x 值
    y0 -- 初始 y 值
    h -- 步长
    num -- 迭代次数
    
    返回：
    x -- x 的数组
    y -- y 的数组
    x1 -- 位移，及速度y对应的积分值
    """
    # 创建 x 和 y 的数组,x1为位移数组
    x = np.zeros(num)
    y = np.zeros(num)
    x1 = np.zeros(num)
    
    # 将初始值放入数组
    x[0] = x0
    x1[0] = 0
    y[0] = y0
    
    # 求解微分方程
    for i in range(1, num):
        y[i] = y[i - 1] + h * f(x[i - 1], y[i - 1])
        x1[i] = x1[i - 1] + y[i] * h
        x[i] = x[i - 1] + h
    
    # 返回结果
    return x, y, x1

# 定义初始条件和计算参数
x0 = 0
y0 = 0
h = 0.1
num = 30

# 使用一级欧拉近似方法求解微分方程
x, y, x1 = euler(f, x0, y0, h, num)

# 绘制位移、速度随图像
plt.plot(x, y, '-o')
plt.title('v-t')  # 绘制图像的标题
plt.xlabel('t')   # 绘制 x 轴的标签
plt.ylabel('v')   # 绘制 y 轴的标签
plt.show()        # 显示图像

plt.plot(x, x1, '-o')
plt.title('x-t')  # 绘制图像的标题
plt.xlabel('t')   # 绘制 x 轴的标签
plt.ylabel('x')   # 绘制 y 轴的标签
plt.show()        # 显示图像
