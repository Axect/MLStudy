import numpy as np
from scipy import var, cov, mean
from scipy.linalg import inv
from numpy import matmul
import matplotlib.pyplot as plt

def sample(n):
    # Gaussian Distribution
    mu_x1, mu_x2 = 2, -2
    mu_y1, mu_y2 = 0, 0

    sig_x = 1
    sig_y = 3
    
    # Pick Samples
    x1, x2 = np.random.normal(mu_x1, sig_x, n), np.random.normal(mu_x2, sig_x, n)
    y1, y2 = np.random.normal(mu_y1, sig_y, n), np.random.normal(mu_y2, sig_y, n)

    # Group
    c1 = (x1, y1)
    c2 = (x2, y2)

    # To Plot
    x = np.append(x1,x2)
    y = np.append(y1,y2)

    return c1, c2, x, y

def main():
    n = 150
    c1, c2, x, y = sample(n)
    (x1, y1) = c1
    (x2, y2) = c2
    
    # Make mean vector & cov matrix
    m1 = np.array(list(map(mean, [x1, y1])))
    m2 = np.array(list(map(mean, [x2, y2])))

    sigma = cov(x1, y1)

    # LDA
    a = matmul(inv(sigma), (m1 - m2))
    b = -0.5 * (matmul(matmul(m1, sigma), m1) - matmul(matmul(m2,sigma),m2))
    print("a=", a)
    print("b=", b)
    
    def L(x):
        return -(a[0] / a[1]) * x - (b / a[1])

    r1, r2 = round(min(x) - 1), round(max(x) + 1)
    lx = np.linspace(r1, r2, num=100)
    ly = [L(x) for x in lx]

    plt.plot(x1,y1,'r.',x2,y2,'b.', lx, ly, 'black')
    plt.xlim([r1,r2])
    plt.ylim([-7,7])
    plt.savefig("lda2.png")

if __name__=="__main__":
    main()
