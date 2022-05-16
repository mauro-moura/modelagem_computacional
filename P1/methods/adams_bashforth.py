import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def adams_bashfort_prof(h, t, y0, y1, N):
    u = np.empty(len(t))
    u[0] = y0
    u[1] = y1

    u0 = np.empty(len(t))
    u0[0] = y0
    u0[1] = y1

    for i in range(1, N):
        # Adams - Bashfort 2 passos
        u0[i+1] = u[i] + 0.5*h*(3* f(t[i], u[i]) - f(t[i - 1], u[i - 1]))
        #u0[i+1] = u0[i] + 0.5*h*(3* f(t[i], u0[i]) - f(t[i - 1], u0[i - 1]))
        # Adams Moulton 1 passo
        u[i + 1] = u[i] + 0.5 * h * (f(t[i + 1], u0[i + 1]) + f(t[i], u[i]))
    
    return u0, u

def adams_bashforth_2(h, x, y0, y1, n):
    y = np.empty(len(t))

    y[0] = y0
    y[1] = y1

    for i in range(0, n - 1):
        y[i + 2] = y[i + 1] + 0.5 * h * (3*f(x[i + 1], y[i + 1]) - f(x[i], y[i]))

    return y

def adams_bashforth_3(h, x, y0, y1, y2, n):
    y = np.empty(len(t))

    y[0] = y0
    y[1] = y1
    y[2] = y2

    for i in range(0, n - 2):
        y[i + 3] = y[i + 2] + h/12 * (23 * f(x[i + 2], y[i + 2]) - 16 * f(x[i + 1], y[i + 1]) + 5 * f(x[i], y[i]))
    
    return y

def adams_bashforth_4(f, h, x, y0, y1, y2, y3, n):
    y = np.zeros(len(x))

    y[0] = y0
    y[1] = y1
    y[2] = y2
    y[3] = y3

    for i in range(0, n - 4):
        y[i + 4] = y[i + 3] + (h/24 * (55 * f(x[i + 3], y[i + 3]) - 59 * f(x[i + 2], y[i + 2]) + 37 * f(x[i + 1], y[i + 1]) - 9 * f(x[i], y[i])))
    
    return y

'''
MODELO PREDITOR CORRETOR
'''
def adams_moulton_1(h, x, y0, y1, n):
    y = np.empty(len(t))
    y[0] = y0
    y[1] = y1

    adams_bashforth_2(h, x, y0, y1, n)

    for i in range(n):
        y[i + 1] = y[i] + h/2 * (f(x[i + 1], y[i + 1]) + f(x[i], y[i]))
    
    return y

def adams_moulton_2(h, x, y0, y1, y2, n):
    y = np.empty(len(t))
    y[0] = y0
    y[1] = y1
    y[2] = y2

    u0 = adams_bashforth_3(h, x, y0, y1, y2, n)

    for i in range(n - 1):
        y[i + 2] = y[i + 1] + h/12 * (5 * f(x[i + 2], u0[i + 2]) + 8 * f(x[i + 1], y[i + 1]) - f(x[i], y[i]))
    
    return y

def adams_moulton_3(f, h, x, y0, y1, y2, y3, n):
    u0 = np.zeros(len(x))
    u0[0] = y0
    u0[1] = y1
    u0[2] = y2
    u0[3] = y3
    
    y = np.zeros(len(x))
    y[0] = y0
    y[1] = y1
    y[2] = y2
    y[3] = y3

    for i in range(n - 3):
        # Preditor
        u0[i + 3] = y[i + 2] + h/24 * (55 * f(x[i + 2], y[i + 2]) - 59 * f(x[i + 1], y[i + 1]) + 37 * f(x[i], y[i]) - 9 * f(x[i - 1], y[i - 1]))
        # Corretor
        y[i + 3] = y[i + 2] + h/24 * (9 * f(x[i + 3], u0[i + 3]) + 19 * f(x[i + 2], y[i + 2]) - 5 * f(x[i + 1], y[i + 1]) + f(x[i], y[i]))

    return y

if __name__ == '__main__':
    import sys
    sys.path.insert(0, './')

    from runge_kutta import RK4
    from modules.equations import Eq_Aula

    equation = Eq_Aula
    f = equation.f
    analitica = equation.analitica

    b = 2 
    a = 0
    N = 10
    y0 = 1
    
    h = (b - a) / N
    t = np.arange(a, b + h, h)

    y = analitica(t)

    y_rk = RK4(f, N, x_i = 0, y_i = y0, a = a, b = b)

    y1 = y_rk[1]
    y2 = y_rk[2]
    y3 = y_rk[3]

    y_rk = np.insert(y_rk, -1, y[-1])

    #u = adams_bashforth_2(h, t, y0, y1, N)
    #u = adams_bashforth_3(h, t, y0, y1, y2, N)
    #u = adams_bashforth_4(h, t, y0, y1, y2, y3, N)

    #u = adams_moulton_1(h, t, y0, y1, N)
    #u = adams_moulton_2(h, t, y0, y1, y2, N)
    u = adams_moulton_3(h, t, y0, y1, y2, y3, N)
    
    u0, _ = adams_bashfort_prof(h, t, y0, y1, N)

    d = {
        'Tempo': t,
        'Bash': u,
        'RK4': y_rk,
        'Bash Prof': u0,
        'Exata': y
    }
    df = pd.DataFrame(d)
    print(df)

    #print(f'{t} \t {u0} \t {u} \t {y}')

    plt.plot(t, u0, 'rv-', label = 'Adams Bash')
    plt.plot(t, u, 'bo-', label = 'Adams Bash Mauro')
    plt.plot(t, y, 'g--', label = 'Exata')
    plt.plot(t, y_rk, label = 'RK4')
    plt.legend()
    plt.show()
