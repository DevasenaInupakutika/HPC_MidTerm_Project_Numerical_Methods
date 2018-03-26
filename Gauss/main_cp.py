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
    for i in range(n):
        pivot(m, n, i)
        for j in range(i+1, n):
            m[j] = [m[j][k] - m[i][k]*m[j][i]/m[i][i] for k in range(n+1)]

    if m[n-1][n-1] == 0: raise ValueError('No unique solution')

    # backward substitution
    x = [0] * n
    for i in range(n-1, -1, -1):
        s = sum(m[i][j] * x[j] for j in range(i, n))
        print("m[i][i] = \n",m[i][i])
        x[i] = (m[i][n] - s) / m[i][i]
    return x

'''
# shorter way to pivot but cannot run in trinket
def pivot(m, n, i):
    max_row = max(range(i, n), key=lambda r: abs(m[r][i]))
    m[i], m[max_row] = m[max_row], m[i]
'''

def pivot(m, n, i):
    global max_row
    max = -1e100
    for r in range(i, n):
        if max < abs(m[r][i]):
             max_row = r
             max = abs(m[r][i])
    m[i], m[max_row] = m[max_row], m[i]

if __name__ == '__main__':
    mat10 = np.random.random((10,11))
    print("Matrix: \n",mat10)
    #Saving array to file
    #np.savetxt('arr1.csv',mat10,delimiter=",")
    list10 = mat10.tolist()
    start = time.clock()
    print(gaussian_elimination_with_pivot(list10))
    finish = time.clock()
    print 'time for solution is ', finish - start,'s'
 