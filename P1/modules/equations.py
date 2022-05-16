import numpy as np

class Eq_1():
    def __init__(self):
        print('Equação: y**(1/3)')

    def f(x, y):
        return y**(1/3)
    
    def df(x, y):
        return 1/(3*(y**(2/3)))
    
    def analitica(t):
        return 2/3 * np.sqrt(2/3*(t**(3/2)))

class Eq_2():
    def __init__(self):
        print('Equação: (1-(4/3*t))*y')

    def f(t, y):
        return (1-(4/3*t))*y
    
    def df(t, y):
        return -(4/3)*y

    def analitica(t):
        return np.exp(t - (2*t**2)/3)

class Eq_Exemplo():
    def __init__(self):
        print('Equação: x-2*y+1')

    def df(t, y):
        return 2

    def f(x, y):
        return x-2*y+1

    def analitica(x) :
        return (1.0/4)*(3*np.exp(-2*x) + 2*x + 1)

class Eq_Trabalho():
    def __init__(self):
        print('Equação: k*(y - T)')

    def f(x, y, k = -0.028, T = 60):
        return k*(y - T)

    def df(t, y):
        return "Tem não"
    
    def analitica(x, T = 60, A = 40) :
        return T + (A*np.exp(-0.028*x))

class Eq_Paper():
    def __init__(self):
        print('Equação: k*(y - T)')

    def f(x, y, k = -0.08, T = 21.1):
        return k*(y - T)

    def df(t, y):
        return "Tem não"
    
    def analitica(x, k = -0.08, T = 21.1, A = 72.2) :
        return T + (A*np.exp(k*x))

class Eq_Aula:
    def f(t, u):
        #return np.sqrt(u + 1)
        return t - u
    
    def analitica(t, u0 = 1):
        #return (t**2 / 4) + t
        return t + 2*np.exp(-t) - 1
