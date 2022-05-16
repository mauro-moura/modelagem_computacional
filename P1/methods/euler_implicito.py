
import numpy as np
import matplotlib.pyplot as plt

from utils import print_error

def euler_implicito(x, y, f, df, max_error = 10**(-10), iterations = 6):
    for i in range(0, n - 1):
        ERRO = 1
        j = 0

        zold = y[i] + h * f(x[i], y[i]) # Xn ou Y[i]

        while ERRO > max_error and j <= iterations:
            F = y[i] + h * f(x[i+1], zold) - zold
            dF = h * df(x[i+1], zold) - 1
            xnew = zold - F/dF
            ERRO = abs(xnew - zold) # Erro Absoluto
            #print(ERRO)
            zold = xnew
            j+=1

        y[i+1] = xnew

    return y

if __name__ == '__main__':
    from equations import Eq_2

    equation = Eq_2
    f = equation.f
    df = equation.df
    analitica = equation.analitica

    n = 100
    a = 0
    b = 1
    y_i = 1
    h  = (b - a) / n

    x = np.zeros(n)
    y = np.zeros(n)
    
    x = np.arange(a, b, h)

    x[0] = 0
    y[0] = y_i

    #z_old = euler_explicito(x, y, f, n, h, y_i = y_i)
    y = euler_implicito(x, y, f, df)

    print_error(x, y, n, analitica)

    plt.figure(1)
    plt.plot(x, y,'red', label='Euler')
    plt.plot(x, analitica(x),'blue',label='Analitica')
    plt.legend()
    plt.show()

