
# coding: utf-8

# In[ ]:

#Generalized Minimal Residual Method (GMRES) Solver Implementation (Non-symmetric matrices)

#Import libraries Numpy, time and sys. Scipy is used to measure the difference in solutions (my implementation and scipy's actual gmres function)
import numpy as np
#For double checking the result using scipy
import scipy.sparse.linalg as spla

import time
import sys
import os

#Results directory
res_dir = "../Results/"

#Generating a row by col matrix for linear system of equations i.e A
def matM_by_N(M,N):
    mat = np.random.random_integers(1,10,size=(M,N))
    return mat

#Generating column vector row by 1 i.e. b
def colN_by_1(row):
    #Column Vector of row elements i.e. b
    col_N = np.random.random_integers(1,10,size=(row,))
    return col_N

#Results Data file Utility (create if not present else append the existing file with all result data)
def res_data_file(M,N,str_result):
    # Create the folder if it doesn't exist
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
        
    # Create an file and write to it in append mode.
    # Open a file with writing mode, and then just close it.
    with open(os.path.join(res_dir, 'GMRES_LSE_'+str(M)+'_by_'+str(N)), 'a') as resfile:
        resfile.write(str_result)
        resfile.close()

#GMRES Solver function        
def GMRes(A, b, x0, e, nmax_iter, restart=None):
    '''
    Parameters:	
        A : {sparse matrix, dense matrix, LinearOperator}

        The real or complex N-by-N matrix of the linear system.

        b : {array, matrix}

        Right hand side of the linear system. Has shape (N,) or (N,1).

        Returns:	
        x : {array, matrix}

        The converged solution.
        
        x0:
        Starting guess for the solution (a vector of zeros by default).
        
        nmax_iter:
        Maximum number of iterations (restart cycles). Iteration will stop after maxiter steps even if the specified tolerance has not been achieved.
        
        restart: int and optional
    '''
    #r = b-Ax residue
    #Computing the solution estimate xk  that minimizes the residual euclidean norm  over a krylov subspace of 
    #dimension k.
    r = b - np.asarray(np.dot(A, x0)).reshape(-1)

    x = []
    q = [0] * (nmax_iter)

    x.append(r)

    q[0] = r / np.linalg.norm(r)

    h = np.zeros((nmax_iter + 1, nmax_iter))

    #Performing the steps of the Arnoldi algorithm using modified Gram-Schmidt 
    for k in range(nmax_iter):
        y = np.asarray(np.dot(A, q[k])).reshape(-1)

        #Arnoldi algorithm 
        for j in range(k):
            h[j, k] = np.dot(q[j], y)
            y = y - h[j, k] * q[j]
            
        #After k steps of Arnoldi, we have upper Hessenberg martrix h and a matrix q whose columns are 
        #orthonormal vectors spanning the Krylov subspace.
        h[k + 1, k] = np.linalg.norm(y)
        if (h[k + 1, k] != 0 and k != nmax_iter - 1):
            q[k + 1] = y / h[k + 1, k]

        b = np.zeros(nmax_iter + 1)
        b[0] = np.linalg.norm(r)
        
        #Solving the least squares problem
        result = np.linalg.lstsq(h, b)[0]
       
        res = np.dot(np.asarray(q).transpose(), result) + x0
        #x.append(res)

    return res


# In[ ]:

#Number of rows M to be entered by user from command line as first argument
M = int(sys.argv[1])

#Number of columns N to be entered by user from command line as second argument
N = int(sys.argv[2])

#Matrix A (M by N) generated with random integers using a python utility defined above
A = matM_by_N(M,N)

#RHS vector (N,) generated with random integers using a python utility defined above
b = colN_by_1(N)

#Starting guess for the solution vector x (a vector of zeros by default)
x0 = np.zeros(shape=(N,))

#Tolerance to achieve but not used in current implementation of GMRES
e = 0

#Maximum number of iterations. Iteration will stop after maxiter steps. (entered as third command line argument while running the program)
nmax_iter = int(sys.argv[3])

#Start the timer: for measuring the time of execution and getting the solution vector
start = time.clock()

#String representing number of iterations
str_iter = "Maximum number of iterations: "+str(nmax_iter)
#Calls the GMRES function and stores the result in x which is a solution vector
x = GMRes(A, b, x0, e, nmax_iter)
str_x = "Solution Vector is: ", x
print(str_x)

#Below one is to cross-check with the Scipy inbuilt implementation and my implementation as part of the project
print("Scipy result: \n",spla.gmres(A,b,x0))

#Below steps are for measuring the error by computing L2 norm (residual norm error)
error_sum = 0
Ax = np.matmul(A,x)
print("Ax is: ",Ax)

#Computing Solution error by calculating the L2 norm
for i in range(len(x)):
    error_sum = error_sum + np.power((b[i] - Ax[i]),2)

str_error_sum = "Residual Norm error is: ", np.sqrt(error_sum) 
print(str_error_sum)

#Measures time until this step and prints it
finish = time.clock()
runtime = finish - start

str_res_runtime = 'Time for solution is ', runtime,'s'
print(str_res_runtime)

#Combined result data
str_result = str(str_x)+'\n'+str(str_error_sum)+'\n'+str(str_res_runtime)+'\n'+str_iter

#Writing the size of linear system of equations (M by N), Solution vector, Error, Run time and Number of iterations to the results file
res_data_file(M,N,str_result)

