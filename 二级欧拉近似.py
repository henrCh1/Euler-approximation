# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 15:40:24 2023

@author: 86319
"""

import numpy as np
import matplotlib.pyplot as plt

def euler_2(f, y0, t0, tn, h):
    """
    使用二级欧拉近似方法求解常微分方程

    参数：
    f：函数，表示微分方程dy/dt=f(t,y)
    y0：浮点数，表示初始条件y(t0)=y0
    t0：浮点数，表示初始时间
    tn：浮点数，表示终止时间
    h：浮点数，表示时间步长

    返回值：
    t：一维NumPy数组，表示时间向量
    y：一维NumPy数组，表示数值解向量
    """
    # 计算时间步数
    N = int((tn - t0) / h)

    # 初始化时间和解向量
    t = np.linspace(t0, tn, N+1)
    y = np.zeros(N+1)
    y[0] = y0

    # 迭代计算解向量
    for i in range(N):
        # 使用二级欧拉近似方法
        y[i+1] = y[i] + h * f(t[i] + h/2, y[i] + h/2 * f(t[i], y[i]))

    return t, y

# 定义微分方程dy/dt=f(t,y)
def f(t, y):
    return -0.02*y/0.05-9.81*np.sin(t)/1.016

# 求解微分方程y' = y - t^2 + 1，y(0) = 0的数值解
t, y = euler_2(f, 0, 0, 25, 0.2)

# 输出数值解
print("数值解：", y)

# 绘制数值解
plt.plot(t, y, '-o')
plt.xlabel('t')
plt.ylabel('y')
plt.title("Solution of y' = y - t^2 + 1 with y(0) = 0")
plt.show()
