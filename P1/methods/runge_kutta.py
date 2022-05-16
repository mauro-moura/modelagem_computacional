import numpy as np

import sys
sys.path.insert(0, './')
from modules.utils import generate_xy

def RK2(f, n, x_i = 0, y_i = 1, a = 0, b = 1):
    x, y, h = generate_xy(n, a, b)
    x[0] = x_i
    y[0] = y_i

    for i in range(n - 1):
        F1 = h * f(x[i], y[i])
        F2 = h * f(x[i + 1], y[i] + F1)
        y[i + 1] = y[i] + (1/2) * F1 + (1/2) * F2
    
    return y

def euler_melhorado(f, n, x_i = 0, y_i = 1, a = 0, b = 1,  w1 = 0, w2 = 1):
    if (w1 + w2 != 1): return print("Os pesos estão errados")
    
    x, y, h = generate_xy(n, a, b)
    x[0] = x_i
    y[0] = y_i

    alpha = beta = 1/(2*w2)

    for i in range(n - 1):
        F1 = h * f(x[i], y[i])
        F2 = h * f(x[i]+alpha*h, y[i] + beta*F1)
        y[i + 1] = y[i] + w1 * F1 + w2 * F2
    
    return y

def RK3(f, n, x_i = 0, y_i = 1, a = 0, b = 1):
    x, y, h = generate_xy(n, a, b)
    x[0] = x_i
    y[0] = y_i

    for i in range(0, n - 1):
        k1 = f(x[i], y[i])
        k2 = f(x[i] + h/2, y[i] + h/2 * k1)
        k3 = f(x[i] + h, y[i] - h * k2 + 2*h*k2)

        y[i + 1] = y[i] + (1/6) * h * (k1 + 4*k2 + k3)
    
    return y

def RK4(f, n, x_i = 0, y_i = 1, a = 0, b = 1):
    x, y, h = generate_xy(n, a, b)
    x[0] = x_i
    y[0] = y_i

    for i in range(0, n - 1):
        k1 = h * f(x[i], y[i])
        k2 = h * f(x[i] + h/2, y[i] + k1/2)
        k3 = h * f(x[i] + h/2, y[i] + k2/2)
        k4 = h * f(x[i] + h, y[i] + k3)

        y[i + 1] = y[i] + (1/6) * (k1 + 2*k2 + 2*k3 + k4)
    
    return y

if __name__ == '__main__':
    print("Oi")
