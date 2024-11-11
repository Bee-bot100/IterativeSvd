import numpy as np

def power_iteration(A, num_iterations: int):
    #Random vector
    b_k = np.random.rand(A.shape[1])

    for _ in range(num_iterations):
        # calculate the matrix-by-vector product Ab
        b_k1 = np.dot(A, b_k)

        # calculate the norm
        b_k1_norm = np.linalg.norm(b_k1)

        # re normalize the vector
        b_k = b_k1 / b_k1_norm
    if b_k[0] < 0:
        b_k = -1 * b_k
    return b_k

def deflation(A,v):
    #using schur deflation
    m_v = np.dot(v,np.transpose(v))
    Am = A - np.dot(A,np.dot(m_v,A))/(np.dot(v,np.dot(A, np.transpose(v))))
    return Am


def giatrikidi(A,v):
    x =  np.dot(np.transpose(v),np.dot(A , v))
    return x

def iterative_svd(A):
    A1 = np.dot(A, np.transpose(A))
    A2 = np.dot(np.transpose(A), A)
    Q = []
    Pt = []
    Z1 = []
    Z2 = []

    for _ in range(len(A1)):
        v1 = power_iteration(A1,10000)
        if giatrikidi(A1,v1) >= 0:
            Z1.append(giatrikidi(A1,v1))
            Q.append(v1)
        A1 = deflation(A1,v1)


    for _ in range(len(A2)):
        v2 = power_iteration(A2,10000)
        if giatrikidi(A2,v2) >= 0:
            Pt.append(v2)
            Z2.append(giatrikidi(A2,v2))
        A2 = deflation(A2,v2) 

    Z1 = np.round(np.sqrt(Z1), 2)
    Z2 = np.round(np.sqrt(Z2), 2)
    Q = np.transpose(np.round(Q, 2))
    Pt = np.round(Pt,2)

    print(f"Q = \n{Q}\nZ1 = \n{Z1}\nZ2 = \n{Z2}\nPT =  \n{Pt}")


A = np.array([[3,1,2], [2, 1 ,2]])
iterative_svd(A)

