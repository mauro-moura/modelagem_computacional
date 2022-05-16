
import sys
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from modules import utils
from methods import adams_bashforth, euler_explicito, runge_kutta

class Eq_P1_S_Cat():
    def __init__(self):
        print('Equação da P1 sem adição de catalisador')

    def f(t, y):
        eq = -(k * (y**3))
        
        #print(f"f {eq} Method {method} y {y}")
        #if (np.isnan(eq)): sys.exit()
        
        return eq

    def analitica(t) :
        return 1 / np.sqrt(2*k*t + (1/(c0**2)))
    
    def GP(t):
        return (2 * k * (c0 ** 2) * t) + 1

class Eq_P1_C_Cat():
    def __init__(self):
        print('Equação da P1 com adição de catalisador')

    def f(t, y):
        eq = -(k_l * (y**2))
        
        #print(f"f {eq} Method {method} y {y}")
        #if (np.isnan(eq)): sys.exit()
        
        return eq
    
    def analitica(t) :
        return 1 / (k_l * t + (1/c0))
    
    def GP(t):
        return (k_l * c0 * t) + 1

def run_graph(n):
    # Graph
    F_GP = equation.GP
    GP = F_GP(x)
    
    plt.plot(x, GP, label='GP')
    plt.title('Valores de GP')
    plt.ylabel('1 / (1-p)²')
    plt.ylabel('GP')
    plt.xlabel('Tempo em minutos')
    plt.savefig(f'{outputs_folder}GP_fig_n{n}.png')
    #plt.show()
    plt.close()

def run_graph_2(n):
    # Graph
    F_GP = equation.GP
    GP = F_GP(x)
    GP_2 = np.sqrt(GP)

    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.plot(x, GP, 'g-')
    ax2.plot(x, GP_2, 'b-')

    plt.title(f'Valores de GP para h={h}')
    plt.ylabel('1 / (1 - p)²')
    plt.ylabel('GP')
    plt.xlabel('Tempo em minutos')

    plt.grid()

    ax1.set_xlabel('Tempo em minutos')
    ax1.set_ylabel('1 / (1-p)²', color='g')
    ax2.set_ylabel('GP', color='b')

    plt.savefig(f'{outputs_folder}GP_fig_n{n}.png')
    #plt.show()
    plt.close()

def run(n):
    global method
    method = "Explicito"
    y_explicito = euler_explicito.euler_explicito(f, n, x_i, y_i, a, b)
    method = "RK 4"
    y_rk4 = runge_kutta.RK4(f, n, x_i, y_i, a, b)
    
    y1 = y_rk4[1]
    y2 = y_rk4[2]
    y3 = y_rk4[3]
    
    method = "adams_bashforth_4"
    y_bash = adams_bashforth.adams_bashforth_4(f, h, x, y_i, y1, y2, y3, n)
    method = "adams_moulton_3"
    y_moulton = adams_bashforth.adams_moulton_3(f, h, x, y_i, y1, y2, y3, n)
    
    y_anal = analitica(x)
    
    erro_rel_explicito = utils.erro_relativo(y_anal, y_explicito)
    erro_rel_rk4 = utils.erro_relativo(y_anal, y_rk4)
    erro_rel_bash = utils.erro_relativo(y_anal, y_bash)
    erro_rel_moulton = utils.erro_relativo(y_anal, y_moulton)
    
    d = {
        "Analítica": y_anal,
        "Euler Explícito": y_explicito,
        "RK4": y_rk4,
        "Adams Bashfort": y_bash,
        "Adams Moulton": y_moulton,
        "Erro Euler Explícito": erro_rel_explicito,
        "Erro RK4": erro_rel_rk4,
        "Erro Bashfort": erro_rel_bash,
        "Erro Preditor Corretor": erro_rel_moulton
    }

    df = pd.DataFrame(d, index = x)
    df.to_excel(f'{outputs_folder}outputs_n{n}.xlsx')
    
    plt.plot(x, y_anal,'green',label='Analitica')
    plt.plot(x, y_explicito,'red',label='Euler Explícito')
    plt.plot(x, y_rk4,'black',label='RK4')
    plt.plot(x, y_bash,'purple',label='Adams Bashfort')
    plt.plot(x, y_moulton,'blue',label='Adams Preditor Corretor')
    plt.title(f'Concentração x Tempo para h={h}')
    plt.ylabel('c(t)')
    plt.xlabel('t')
    plt.legend()
    plt.savefig(f'{outputs_folder}figure_n{n}.png')
    #plt.show()
    plt.close()

if __name__ ==  '__main__':
    i_number = 1

    outputs_folder = ['./outputs_p1/com_catalisador/', './outputs_p1/sem_catalisador/']
    outputs_folder = outputs_folder[i_number]

    equation = [Eq_P1_C_Cat, Eq_P1_S_Cat]
    equation = equation[i_number]

    f = equation.f
    analitica = equation.analitica

    c0 = 2 # mol / l
    #k_l = 60.0 * 10**-3 # l / mol s
    k_l = 0.06
    #k = ((60 ** 2) * 10**-5) # l² / mol² s²
    k = 0.036
    
    a = 0
    b = 800
    x_i = 0
    y_i = c0

    #n_list = [100, 800, 1200, 1600, 3200]
    n_list = [1000]

    for n in n_list:
        h  = (b - a) / n
        
        #y = np.zeros(n)
        x = np.arange(a, b, h)

        print(f"K {k} K' {k_l}")
        
        #y[0] = y_i
        x[0] = x_i
        
        run(n)

        if (i_number == 0):
            run_graph(n)
        else:
            run_graph_2(n)