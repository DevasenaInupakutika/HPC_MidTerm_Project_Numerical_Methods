import time
from matrix import height

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
        x[i] = (m[i][n] - s) / m[i][i]
    return x

'''
# shorter way to pivot but cannot run in trinket
def pivot(m, n, i):
    max_row = max(range(i, n), key=lambda r: abs(m[r][i]))
    m[i], m[max_row] = m[max_row], m[i]
'''

def pivot(m, n, i):
    max = -1e100
    for r in range(i, n):
        if max < abs(m[r][i]):
             max_row = r
             max = abs(m[r][i])
    m[i], m[max_row] = m[max_row], m[i]

if __name__ == '__main__':
    #m = [[0,-2,6,-10], [-1,3,-6,5], [4,-12,8,12]]
    #m = [[1,-1,3,2], [3,-3,1,-1], [1,1,0,3]]
    m = [[1,-1,3,2], [6,-6,2,-2], [1,1,0,3]]
    mat10 = [[ 0.41631682,0.41874375,  0.06890374,  0.27923266,  0.71359717,  0.55799046
   ,0.47003857,  0.06417241 , 0.75248972 , 0.66811458,0.5],
   [ 0.235443, 0.78181228,  0.26476632,  0.74943899,  0.06038605,  0.2325526,
   0.24081837,  0.88391347,  0.60439319,  0.4555491,0.6 ],
   [ 0.20362203,  0.07310563,  0.04263453,  0.95754093 , 0.21495986 , 0.67428786,
   0.59842509 , 0.96378373 , 0.81944256 , 0.12560601,0.3],
   [ 0.66585332 , 0.58816975 , 0.0957221  , 0.1840348 ,  0.7220652 ,  0.45054097,
   0.21754685 , 0.36792945 , 0.50677586 , 0.96329777,0.7],
   [ 0.75927944 , 0.97806352 , 0.86934374 , 0.40145078,  0.98782955,  0.99483838,
   0.81053141 , 0.76083168 , 0.920159, 0.63028815,0.1],
   [ 0.71318319 , 0.87396693 , 0.32567136 , 0.21466519,  0.21047783,  0.70103555,
   0.26506325 , 0.32285287 , 0.68788252 , 0.43491421,0.2],
   [ 0.4198022  , 0.5351444  , 0.20005959 , 0.69666804,  0.96278402,  0.90388817,
   0.91515186 , 0.26688272 , 0.51832622 , 0.53362616,0.5],
   [ 0.77811618 , 0.33484869 , 0.52769649 , 0.42318873,  0.64527922 , 0.99097465,
   0.38169553 , 0.18147161 , 0.42962635 , 0.84029842,0.9],
   [ 0.26232138 , 0.35753828 , 0.24328579 , 0.37643709,  0.21039178 , 0.87194096,
   0.95990441 , 0.21049965 , 0.20836219 , 0.42037664,1.0],
   [ 0.90901387 , 0.88163604 , 0.7410935  , 0.98025649,  0.73659163,  0.27336471,
   0.72174938 , 0.0526961  , 0.73706415 , 0.88463602,0.6]]            
    print(gaussian_elimination_with_pivot(m))
    start = time.clock()
    print(gaussian_elimination_with_pivot(mat10))
    finish = time.clock()
    print 'time for solution is ', finish - start,'s'
    """  
    m = [[4,4,0,400], [-1,4,2,400], [0,-2,4,400]]   # aj Montri p80  [50, 50, 125]
    print(gaussian_elimination_with_pivot(m))
    """

# gaussian elimination with pivot
# author: Worasait Suwannik
# date: May 2015

