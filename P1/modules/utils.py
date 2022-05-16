import numpy as np
import matplotlib.pyplot as plt

def generate_xy(n, a = 0, b = 1):
    y = np.zeros(n)
    x = np.zeros(n)
    h  = (b - a) / n
    x = np.arange(a, b, h)
    #print(x)
    return x, y, h

def sol_analitica(x, n, analitica):
    y_sol = np.zeros(n)
    for i in range(n):
        y_sol[i] = analitica(x[i])
    return y_sol

def print_error(x, y, n, analitica):
    y_anal = sol_analitica(x, n, analitica)
    erro = erro_relativo(y_anal, y)
    print("\t  \t  ")
    for i in range(n):
        print(f'Passo: {x[i]:.03f} \t Euler: {y[i]:.03f} \t Analitica: {y_anal[i]:.03f} \t Erro Relativo: {erro[i]:.03f}')

def erro_absoluto(y_true, y_pred):
    return abs(y_true - y_pred)

def erro_relativo(y_true, y_pred):
    return 100* (abs(y_true - y_pred) / y_true)

def plot_error(x, y, analitica = None, save = False, name='fig'):
    plt.figure(1)
    plt.plot(x, y,'red',label='Euler')
    if (analitica): plt.plot(x, analitica(x),'blue', label='Analitica')
    plt.legend()
    if (save): plt.savefig(name + '.png')
    plt.show()
