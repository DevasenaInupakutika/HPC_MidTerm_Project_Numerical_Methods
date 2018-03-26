import numpy as np
import time
import sys

#Generating a row by col matrix for linear system of equations i.e A
def mat10_by_10(row,col):
    mat10 = np.random.random_integers(1,10,size=(row,col))
    #mat10 = np.asmatrix(mat10)
    return mat10

#Generating column vector row by 1 i.e. b
def col10_by_1(row):
    #Column Vector of row elements i.e. b
    col_10 = np.random.random_integers(1,10,size=(row,1))
    #col_10 = np.asmatrix(col_10)
    return col_10
  
#Pass iteration as third command line argument to program execution
ITERATION_LIMIT = int(sys.argv[3])

#Pass row number as first command line argument to program execution (system of linear equations)
row = int(sys.argv[1])

#Pass column number as second command line argument to program execution
col = int(sys.argv[2])

mat1 = mat10_by_10(row,col) #[[10.0,-1.0,2.0,0.0],[-1.0,-11.0,-1.0,3.0],[2.0,-1.0,10.0,-1.0],[0.0,3.0,-1.0,8.0]]
col1 = col10_by_1(row) #[6.0,25.0,-11.0,15.0]
#INITIALISE THE MATRIX
A = np.array(mat1)

#INITIALISE THE RHS VECTOR
b = np.array(col1)

#PRINTS THE SYSTEM
print("System:")

start = time.clock()

for i in range(A.shape[0]):
    row = ["{}*x{}".format(A[i,j],j+1) for j in range(A.shape[1])]
    print(" + ".join(row),"=",b[i])
    
print()

x = np.zeros_like(b)

for it_count in range(ITERATION_LIMIT):
    #print("Current solution:",x)
    x_new = np.zeros_like(x)
    
    
    for i in range(A.shape[0]):
        s1 = np.dot(A[i,:i],x[:i])
        s2 = np.dot(A[i,i+1:],x[i+1:])
        x_new[i] = (b[i] - s1 - s2) / A[i, i]
        
        
    if np.allclose(x,x_new,atol=1e-10,rtol=0.0):
        break
        
    x = x_new
    
print("Solution: ")

print(x)

error = np.dot(A,x) - b
print("Error:")
print(error)

error_sum = 0
Ax = np.matmul(A,x)
print("Ax is: ",Ax)

#Computing Solution error by calculating the L2 norm
for i in range(len(x)):
    error_sum = error_sum + np.power((b[i] - Ax[i]),2)
    
print("Norm error is: \n",np.sqrt(error_sum))

finish = time.clock()

print('time for solution is ', finish - start,'s')
    