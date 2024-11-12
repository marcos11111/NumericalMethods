import numpy as np

def gauss_solve(Mij,bi,x,n):
    Mij=np.array(Mij,copy=True,dtype=float).reshape((n, n))
    bi=np.array(bi,copy=True,dtype=float)
    for i in range(n):
        for j in range(i+1,n):
            if Mij[i][i]==0:
                raise ValueError("Zero on the diagonal, cannot proceed without pivoting")
            factor=Mij[j][i] / Mij[i][i]
            for k in range(i, n):
                Mij[j][k]-=factor*Mij[i][k]
            bi[j]-=factor*bi[i]
    
    for i in range(n-1,-1,-1):
        sum_ax=sum(Mij[i][j]*x[j] for j in range(i+1,n))
        x[i]=(bi[i]-sum_ax)/Mij[i][i]
    
    return x

Mij=[2.0, 0.1, -0.2, 0.05, 4.2, 0.032, 0.12, -0.07, 5.0]
bi=[10, 11, 12]
n=len(bi)
x=[0]*n

# Solve the system
x = gauss_solve(Mij,bi,x,n)
print("Solution x:", x)
