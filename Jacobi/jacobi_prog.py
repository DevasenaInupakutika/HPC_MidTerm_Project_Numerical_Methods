#Jacobi Solver Implementation

#Import statements for different libraries in Python
import numpy as np
import time
import sys
import os

#Results directory
res_dir = "../Results/"

#Generating a row by col matrix for linear system of equations i.e A
def matM_by_N(M,N):
    mat = np.random.random_integers(0.5,1,size=(M,N))
    return mat

#Generating column vector row by 1 i.e. b
def colN_by_1(row):
    #Column Vector of row elements i.e. b
    col_N = np.random.random_integers(0.5,1,size=(row,1))
    return col_N

#Results Data file Utility (create if not present else append the existing file with all result data)
def res_data_file(M,N,str_result):
    # Create the folder if it doesn't exist
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
        
    # Create an file and write to it in append mode.
    # Open a file with writing mode, and then just close it.
    with open(os.path.join(res_dir, 'JACOBI_LSE_'+str(M)+'_by_'+str(N)), 'a') as resfile:
        resfile.write(str_result)
        resfile.close()

#Pass row number as first command line argument to program execution (system of linear equations)
M = int(sys.argv[1])

#Pass column number as second command line argument to program execution
N = int(sys.argv[2])

#Pass iteration as third command line argument to program execution
ITERATION_LIMIT = int(sys.argv[3])
#String representing number of iterations
str_iter = "Maximum number of iterations: "+str(ITERATION_LIMIT)


#Matrix A (M by N) generated with random integers using a python utility defined above
mat1 = matM_by_N(M,N) #[[10.0,-1.0,2.0,0.0],[-1.0,-11.0,-1.0,3.0],[2.0,-1.0,10.0,-1.0],[0.0,3.0,-1.0,8.0]]
#INITIALISE THE MATRIX
A = np.array(mat1)

#RHS vector (N,1) generated with random integers using a python utility defined above
col1 = colN_by_1(M) #[6.0,25.0,-11.0,15.0]
#INITIALISE THE RHS VECTOR
b = np.array(col1)

#PRINTS THE SYSTEM
print("System:")

#Start the timer: for measuring the time of execution and getting the solution vector
start = time.clock()

for i in range(A.shape[0]):
    row = ["{}*x{}".format(A[i,j],j+1) for j in range(A.shape[1])]
    print(" + ".join(row),"=",b[i])
    
print()

#Initialising the solution vector to vector of zeros
x = np.zeros_like(b)

#Jacobi algorithm for determining the solutions of a diagonally dominant system of linear equations
for it_count in range(ITERATION_LIMIT):
    #print("Current solution:",x)
    x_new = np.zeros_like(x)
    
    #Matrix A is split into the diagonal component and remaining component
    #Each diagonal element is solved for and an approximate value is plugged in. 
    #This process is then iterated over (as in outer loop) until it converges.
    for i in range(A.shape[0]):
        s1 = np.dot(A[i,:i],x[:i])
        s2 = np.dot(A[i,i+1:],x[i+1:])
        x_new[i] = (b[i] - s1 - s2) / A[i, i]
        
    #Testing if the 2 numpy arrays x and x_new are close to each other    
    if np.allclose(x,x_new,atol=1e-10,rtol=0.0):
        break
        
    x = x_new
    
str_x = "Solution Vector is: ", x
print(str_x)

'''
error = np.dot(A,x) - b
print("Error:")
print(error)
'''
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
    