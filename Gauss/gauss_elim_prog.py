
# coding: utf-8

# In[ ]:

#Import statements for different libraries in Python
import time
from matrix import height
import numpy as np
import sys

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
    

#Generating a row by col matrix for linear system of equations i.e A
def mat10_by_10(row,col):
    mat10 = np.random.random((row,col))
    #mat10 = np.asmatrix(mat10)
    return mat10

#Generating column vector row by 1 i.e. b
def col10_by_1(row):
    #Column Vector of row elements i.e. b
    col_10 = np.random.random((row,1))
    #col_10 = np.asmatrix(col_10)
    return col_10
    
if __name__ == '__main__':
    
    row = int(sys.argv[1])
    col = int(sys.argv[2])
    
    #A matrix
    mat1_10 = mat10_by_10(row,col)
    print("Matrix A: ",mat1_10)
    
    #b vector
    col1_10 = col10_by_1(row)
    print("Vector b: ",col1_10)
    #print("10 by 10 matrix A is \n",mat1_10)
    #print("Column vector for b is \n",col1_10)

    #Concatenating A and b horizontally to create augmented matrix
    #mat_A_B = np.hstack((mat1_10,col1_10))
    mat_A_B = np.concatenate((mat1_10,col1_10),axis=1)
    l_A_B = mat_A_B.tolist()
    #print("Augmented matrix A_B is \n",mat_A_B)
    print("Size of matrix \n",mat_A_B.shape)
    start = time.clock()
    
    #Calculating the solution
    x_final = gaussian_elimination_with_pivot(l_A_B)
    print("Solution is: \n ",np.array(x_final))
    print("Dimensions of solution: \n",len(x_final))
    
    print("Dimensions of A: \n",len(mat1_10.tolist()))
    print("Dimensions of b: \n",len(col1_10.tolist()))
    
    #Calculating the error
    error = col1_10 - np.dot(mat1_10,x_final)
    print("Error:")
    print(error)
    
    error_sum = 0
    Ax = np.matmul(mat1_10,np.array(x_final))
    
    print("Ax is: ",Ax)
    
    #Computing error by calculating the L2 norm
    for i in range(len(x_final)):
        error_sum = error_sum + (col1_10[i] - Ax[i])**2
    
    print("Norm error is: \n",np.sqrt(error_sum))
    
    finish = time.clock()
    print 'time for solution is ', finish - start,'s'


