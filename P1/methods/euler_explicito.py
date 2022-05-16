import numpy as np
import matplotlib.pyplot as plt

from modules.utils import generate_xy

def euler_explicito(f, n, x_i = 0, y_i = 1, a = 0, b = 1):
    x, y, h = generate_xy(n, a, b)
    for i in range(0, n):
        if (i == 0):
            x[i] = x_i
            y[i] = y_i
        else:
            y[i] = y[i-1] + h * f(x[i-1], y[i-1])
    return y

def euler_explicito_2(f, n, x_i = 0, y_i = 1, a = 0, b = 1):
    x, y, h = generate_xy(n, a, b)

    x[0] = x_i
    y[0] = y_i
    
    for i in range(0, n - 1):
        y[i + 1] = y[i] + h * f(x[i], y[i])
    
    return y

def euler_centrada(f, n, x_i = 0, y_i = 1, a = 0, b = 1):
    x, y, h = generate_xy(n, a, b)
    x[0] = x_i
    y[0] = y_i

    for i in range(1, n):
        euler = y[i - 1] + (h/2) * f(x[i - 1], y[i - 1])
        y[i] = y[i - 1] + h * f(x[i - 1] + h/2, euler)
    
    return y

def euler_ponto_medio(f, n, x_i = 0, y_i = 1, a = 0, b = 1):
    x, y, h = generate_xy(n, a, b)
    x[0] = x_i
    y[0] = y_i

    k1 = f(x[0],y[0])
    y[1] = y[0] + h * k1

    x[1] = x[0] + h
    k2 = f(x[1], y[1])
    y[1] = y[0] + (h/2) * (k1 + k2)

    for i in range(1, n - 1):
        y[i + 1] = y[i - 1] + 2*h*f(x[i], y[i])
    
    return y

if __name__ == '__main__':
    def analitica(x) :
        return (1.0/4)*(3*np.exp(-2*x) + 2*x + 1)
    
    def euler(x, y):
        return x-2*y+1
    
    n = 100
    a = 0
    b = 1
    h  = (b - a) / n
    
    y = np.zeros(n)
    x = np.zeros(n)
    x = np.arange(0.0, 1.0, h)

    y = euler_explicito(x, y, euler, n, h)
    
    plt.figure(1)
    plt.plot(x, y,'red',label='Euler')
    plt.plot(x,analitica(x),'blue',label='Analitica')
    plt.legend()
    plt.show()
