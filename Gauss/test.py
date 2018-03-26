
# coding: utf-8

# In[ ]:

import time
from matrix import height
import numpy as np


def gaussian_elimination_with_pivot(m):
    """
    Parameters
    ----------
    m  : list of list of floats (matrix)

    Returns
    -------  
    list of floats
      solution to the system of linear equation
  
    Raises
    ------
    ValueError
        no unique solution
    """
    # forward elimination
    n = height(m)
    print("n = \n",n)
    for i in range(n):
        pivot(m, n, i)
        for j in range(i+1, n):
            m[j] = [m[j][k] - m[i][k]*m[j][i]/m[i][i] for k in range(n+1)]

    if m[n-1][n-1] == 0: raise ValueError('No unique solution')

    # backward substitution
    x = [0] * n
    for i in range(n-1, -1, -1):
        s = sum(m[i][j] * x[j] for j in range(i, n))
        x[i] = (m[i][n] - s) / m[i][i]
    return x

def pivot(m, n, i):
    max = -1e100
    for r in range(i, n):
        if max < abs(m[r][i]):
             max_row = r
             max = abs(m[r][i])
    m[i], m[max_row] = m[max_row], m[i]
    

#Generating a 100 by 100 matrix for linear system of equations i.e A
def mat100_by_100():
    mat100 = np.random.random((10000,10000))
    mat100 = np.asmatrix(mat100)
    return mat100

#Generating column vector 100 by 1 i.e. b
def col100_by_1():
    #Column Vector of 100 elements i.e. b
    col_100 = np.random.random((10000,1))
    col_100 = np.asmatrix(col_100)
    return col_100
    
if __name__ == '__main__':
    mat1_100 = mat100_by_100()
    col1_100 = col100_by_1()
    #print("100 by 100 matrix A is \n",mat1_100)
    #print("Column vector for b is \n",col1_100)

    #Concatenating A and b horizontally to create augmented matrix
    #mat_A_B = np.hstack((mat1_100,col1_100))
    mat_A_B = np.concatenate((mat1_100,col1_100),axis=1)
    l_A_B = mat_A_B.tolist()
    #print("Augmented matrix A_B is \n",mat_A_B)
    print("Size of matrix \n",mat_A_B.shape)
    start = time.clock()
    print(gaussian_elimination_with_pivot(l_A_B))
    finish = time.clock()
    print 'time for solution is ', finish - start,'s'


