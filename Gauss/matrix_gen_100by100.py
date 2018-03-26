import numpy as np
import myGauss

#Generating a 100 by 100 matrix for linear system of equations i.e A
def mat100_by_100():
    mat100 = np.random.random((100,100))
    mat100 = np.asmatrix(mat100)
    return mat100

#Generating column vector 100 by 1 i.e. b
def col100_by_1():
    #Column Vector of 100 elements i.e. b
    col_100 = np.random.random((100,1))
    col_100 = np.asmatrix(col_100)
    return col_100

mat1_100 = mat100_by_100()
col1_100 = col100_by_1()
print("100 by 100 matrix A is \n",mat1_100)
print("Column vector for b is \n",col1_100)

#Concatenating A and b horizontally to create augmented matrix
mat_A_B = np.hstack((mat1_100,col1_100))
print("Augmented matrix A_B is \n",mat_A_B)

#Storing the generated matrix to file
