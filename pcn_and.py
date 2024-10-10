"""
Neurônio Perceptron para resolver função AND
Victor Passos
9/10/24
"""
import numpy as np
import math
import matplotlib.pyplot as plt


def round(x, decs):
    """
    Função de arredondamento.

    x - valor a ser arredondado;
    decs - número de casas decimais
    """
    if isinstance(x, (float, int, np.float64)):
        return (math.floor((10**decs)*x))/(10**decs)
    m,n = x.shape
    for i in range(m):
        for j in range(n):
            x[i,j] = (math.floor((10**decs)*x[i,j]))/(10**decs)                
    return x   

def signal(x):
    """
    Função degrau

    x - valor de entrada
    """
    if x >= 0:
        return 1
    else:
        return 0


P = np.array([[0,0], [0,1], [1,0], [1,1]])
d = np.array([[0, 0, 0, 1]])
W = np.array([[0, 1]])
b = 0.5
alpha = 0.2
epoch = 1
epochMAX = 500
i = 0
err = 1

while (epoch<=epochMAX):    
    
    total_error = 0

    for i in range(4):
        y = np.matmul(W,P.transpose()[:,i]) + b
        out = signal(y)
        err = d[0,i] - out
        total_error += abs(err)
        
        W = W + alpha * err * P[i,:]
        b = b + alpha * err
    
    if total_error < 1e-5:
        print(f"Convergiu após {epoch} épocas.")
        break
    
    print(f"Epoch: {epoch}, Weights: {round(W, 4)}, Bias: {round(b, 4)}")
    epoch += 1   

for i in range(4):
    y = np.matmul(W, P.transpose()[:, i]) + b
    out = signal(y)
    print(f"Entrada: {P[i,:]}, Saída: {out}")


x1 = np.arange(0, 1.1, 0.1)
x2 = -(W[0,0] / W[0,1]) * x1 - (b / W[0,1])
plt.plot(x1, x2, '-')
plt.scatter(P[:, 0], P[:, 1], color='red', label='Pontos de X')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Semi-Plano Dividido pelo Perceptron')

plt.legend()

plt.show()