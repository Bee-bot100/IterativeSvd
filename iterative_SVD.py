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

def Z_(Q,Z,P):
    rows_Q = Q.shape[1]
    columns_Pt = P.shape[1]
    # Tạo ma trận không vuông toàn số 0 với kích thước mong muốn
    non_square_matrix = np.zeros((rows_Q, columns_Pt))

    #tao ma tran giatrikidi
    for i, value in enumerate(Z):
        if i < rows_Q and i < columns_Pt:
            non_square_matrix[i, i] = value
    return non_square_matrix

def iterative_svd(A):
    A1 = np.dot(A, np.transpose(A))
    A2 = np.dot(np.transpose(A), A)
    Q = []
    Pt = []
    Z1 = []

    #Tim cac cot ma tran Q
    for _ in range(len(A1)):
        v1 = power_iteration(A1,10000)
        if round(giatrikidi(A1,v1)) <= 0:
            break
        Z1.append(giatrikidi(A1,v1))
        Q.append(v1)
        A1 = deflation(A1,v1)

    #Tìm cac hang ma tran Pt
    for _ in range(len(A2)):
        v2 = power_iteration(A2,10000)
        if round(giatrikidi(A2,v2)) <= 0:
            break
        Pt.append(v2)
        A2 = deflation(A2,v2) 

    #Lam tron so
    Z1 = np.round(np.sqrt(Z1), 2)
    Q = np.transpose(np.round(Q, 2))
    P = np.transpose(np.round(Pt,2))

    Z1 = Z_(Q,Z1,P)
    return Q,Z1,P

