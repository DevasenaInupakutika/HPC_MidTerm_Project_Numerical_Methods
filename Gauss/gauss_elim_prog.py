
# coding: utf-8

# In[ ]:

#Gauss Solver Implementation

#Import statements for different libraries in Python
import time
from matrix import height
import numpy as np
import sys
import os

#Results directory
res_dir = "../Results/"

#Gaussian elimination with pivoting main function with the goal of putting the augmented matrix [A b] into the reduced
#row-echelon form
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

#Pivot Function to make an element above and below a leading one to be zero
#Pick a column which has mostly zeros in it. Pivoting on a 1 if possible
def pivot(m, n, i):
    max = -1e100
    for r in range(i, n):
        if max < abs(m[r][i]):
             max_row = r
             max = abs(m[r][i])
    m[i], m[max_row] = m[max_row], m[i]
    

#Generating a row by col matrix for linear system of equations i.e A
def matM_by_N(row,col):
    mat = np.random.random((row,col))
    return mat

#Generating column vector row by 1 i.e. b
def colM_by_1(row):
    #Column Vector of row elements i.e. b
    col = np.random.random((row,1))
    return col

#Results Data file Utility (create if not present else append the existing file with all result data)
def res_data_file(M,N,str_result):
    # Create the folder if it doesn't exist
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
        
    # Create an file and write to it in append mode.
    # Open a file with writing mode, and then just close it.
    with open(os.path.join(res_dir, 'GAUSS_LSE_'+str(M)+'_by_'+str(N)), 'a') as resfile:
        resfile.write(str_result)
        resfile.close()
        
if __name__ == '__main__':
    
    #Number of rows to be entered by user from command line as first argument
    row = int(sys.argv[1])
    
    #Number of columns to be entered by user from command line as second argument
    col = int(sys.argv[2])
    
    #A matrix
    mat1 = matM_by_N(row,col)
    print("Matrix A: ",mat1)
    
    #b vector RHS Vector
    col1 = colM_by_1(row)
    print("Vector b: ",col1)
    
    #print("10 by 10 matrix A is \n",mat1_10)
    #print("Column vector for b is \n",col1_10)

    #Concatenating A and b horizontally to create augmented matrix
    #mat_A_B = np.hstack((mat1,col1))
    mat_A_B = np.concatenate((mat1,col1),axis=1)
    
    l_A_B = mat_A_B.tolist()  #Converting the numpy array to list of list of floats
    #print("Augmented matrix A_B is \n",mat_A_B)
    print("Size of matrix \n",mat_A_B.shape)
    
    #Start the timer: for measuring the time of execution and getting the solution vector
    start = time.clock()
    
    #Calculating the solution
    x_final = gaussian_elimination_with_pivot(l_A_B)
    str_x = "Solution Vector is: ", np.array(x_final)
    print(str_x)
    
    print("Dimensions of solution: \n",len(x_final))
    
    print("Dimensions of A: \n",len(mat1.tolist()))
    print("Dimensions of b: \n",len(col1.tolist()))
    
    '''
    #Calculating the error
    error = col1 - np.dot(mat1,x_final)
    print("Error:")
    print(error)
    '''
    #Below steps are for measuring the error by computing L2 norm (residual norm error)
    error_sum = 0
    Ax = np.matmul(mat1,np.array(x_final))
    print("Ax is: ",Ax)
    
    #Computing error by calculating the L2 norm
    for i in range(len(x_final)):
        error_sum = error_sum + (col1[i] - Ax[i])**2
        
    str_error_sum = "Residual Norm error is: ", np.sqrt(error_sum) 
    print(str_error_sum)
    
    #Measures time until this step and prints it
    finish = time.clock()
    runtime = finish - start

    str_res_runtime = 'Time for solution is ', runtime,'s'
    print(str_res_runtime)
    
    #Combined result data
    str_result = str(str_x)+'\n'+str(str_error_sum)+'\n'+str(str_res_runtime)

    #Writing the size of linear system of equations (row by col), Solution vector, Error and Run time to the results file
    res_data_file(row,col,str_result)


