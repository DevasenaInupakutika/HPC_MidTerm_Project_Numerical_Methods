import numpy as np
import myGauss

#Generating a 10 by 10 matrix for linear system of equations i.e A
def mat10_by_10():
    mat10 = np.random.random((10,10))
    mat10 = np.asmatrix(mat10)
    return mat10

#Generating column vector 10 by 1 i.e. b
def col10_by_1():
    #Column Vector of 10 elements i.e. b
    col_10 = np.random.random((10,1))
    col_10 = np.asmatrix(col_10)
    return col_10

mat1_10 = mat10_by_10()
col1_10 = col10_by_1()
print("10 by 10 matrix A is \n",mat1_10)
print("Column vector for b is \n",col1_10)

#Concatenating A and b horizontally to create augmented matrix
mat_A_B = np.hstack((mat1_10,col1_10))
print("Augmented matrix A_B is \n",mat_A_B)

#Storing the generated matrix to file
