
import numpy as np

def newton(y_i, n, f, df):
    y = np.zeros(n)

    for i in range(n):
        if (i == 0):
            y[i] = y_i
        else:
            y[i] = y[i - 1] - f(y[i - 1])/(df(y[i -1]))
    print("Y_Final", y[-1])
    return y

if __name__ == '__main__':
    def f(x):
        return x**2 - np.exp(x)

    def df(x):
        return 2*x - np.exp(x)

    y_i = -0.5
    n = 10
    #x = np.zeros(n)

    y = newton(y_i, n, f, df)

    print(y)
