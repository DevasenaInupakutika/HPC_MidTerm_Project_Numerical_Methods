
# coding: utf-8

# In[ ]:

#Preconditioned Conjugate Gradient (PCG) Solver Implementation (Symmetric Matrices)

#Import statements for different libraries in Python
import numpy as np
import sys
import time
import os

#Results directory
res_dir = "../Results/"

#Generating column vector row by 1 i.e. b
def colN_by_1(row):
    #Column Vector of row elements i.e. b
    col_N = np.random.random_integers(1,10,size=(row,1))
    return col_N

#Results Data file Utility (create if not present else append the existing file with all result data)
def res_data_file(N,str_result):
    # Create the folder if it doesn't exist
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
        
    # Create an file and write to it in append mode.
    # Open a file with writing mode, and then just close it.
    with open(os.path.join(res_dir, 'PCG_LSE_'+str(N)+'_by_'+str(N)), 'a') as resfile:
        resfile.write(str_result)
        resfile.close()

#Preconditioned Conjugate Gradient Descent Function
def pcg(A,b,x0,TOLERANCE, MAX_ITERATIONS,M):
    """
    A function to solve [A]{x} = {b} linear equation system with the 
    preconditioned conjugate gradient method.
    More at: https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_preconditioned_conjugate_gradient_method
    ========== Parameters ==========
    A : array 
        A real symmetric positive definite matrix.
    b : vector
        The right hand side (RHS) vector of the system.
    x0 : vector
        The starting guess for the solution.
    MAX_ITERATIONS : integer
        Maximum number of iterations. Iteration will stop after maxiter 
        steps even if the specified tolerance has not been achieved.
    TOLERANCE : float
        Tolerance to achieve. The algorithm will terminate when either 
        the relative or the absolute residual is below TOLERANCE.
    M: {sparse matrix, dense matrix, LinearOperator}
       Preconditioner for A. The preconditioner should approximate the inverse of A. 
       Effective preconditioning dramatically improves the rate of convergence, 
       which implies that fewer iterations are needed to reach a given error tolerance.    
    """
    #   Initializations  
    M_inv = np.linalg.inv(M) #Inverse of the preconditioner 
    
    #Initialise the solution vector to the guess values
    x = x0 
    i = 0
    
    #Residue b-Ax
    r = b - np.dot(A, x)
    
    #Preconditioning (Matrix M) is implemented in this solver for conjugate gradient (CG) to ensure convergence of CG
    #Standard method is to multiply M_inv with the Ax = b equation. It is chosen such that it is better conditioned than 
    # A
    #The preconditioner should approximate the inverse of A. Effective preconditioning dramatically improves the 
    #rate of convergence, which implies that fewer iterations are needed to reach a given error tolerance.
    z = np.matmul(M_inv,r)
    p = z
    r_k_norm = np.dot(r.T,p)
   
    #Start iterations   
    while i < MAX_ITERATIONS:
        Ap = np.dot(A,p)
        alpha = r_k_norm / np.dot(p.T,Ap)
        x += alpha * p
        r += alpha * Ap
        r_kplus1_norm = np.dot(r.T,p)
        beta = r_kplus1_norm / r_k_norm
        r_k_norm = r_kplus1_norm
        
        #The algorithm terminates when either the relative or the absolute residual is below TOLERANCE
        if r_kplus1_norm < TOLERANCE:
            print 'Itr: ',i
            break
        p = p + beta * p 
        i+=1
    return x
        


if __name__ == '__main__':

    #First command line parameter is dimension of a symmetric matrix and second command line parameter is maximum iterations
    #Generating symmetric A matrix
    N = int(sys.argv[1])
    
    #Maximum number of iterations to be entered as second command line argument
    MAX_ITERATIONS = int(sys.argv[2])
    #String representing number of iterations
    str_iter = "Maximum number of iterations: "+str(MAX_ITERATIONS)
    
    #RHS Vector
    #Generating a vector b depending on size N entered by user i.e. INITIALISE THE RHS VECTOR
    b = colN_by_1(N)
    
    #Creating a symmetric matrix of size N by N
    A = np.random.random_integers(1,10,size=(N,N))
    A_symm = (A + A.T)/2
    print("Symmetric matrix A is: \n",A_symm)
    print("Vector b is: \n",b)
    
    
    #Calling preconditioned conjugate gradient function
    
    #Tolerance to achieve. The algorithm will terminate when either the relative or the absolute residual is below TOLERANCE
    TOLERANCE = 1.0e-05
    
    #Starting guess for the solution x0
    x0 = np.zeros(shape=(N,1))
    print("Initial guess: \n",x0)
    
    #Start the timer: for measuring the time of execution and getting the solution vector
    start = time.clock()

    #Solution vector with preconditioner M as approximation of inverse of A matrix
    x_final = pcg(A_symm,b,x0,TOLERANCE,MAX_ITERATIONS,A_symm)
    str_x = "Solution Vector is: ", x_final
    print(str_x)
    
    #Below steps are for measuring the error by computing L2 norm (residual norm error)
    error_sum = 0
    Ax = np.matmul(A_symm,x_final)
    print("Ax is: ",Ax)

    #Computing Solution error by calculating the L2 norm
    for i in range(len(x_final)):
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
    res_data_file(N,str_result)
    
    
    



