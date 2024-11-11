import numpy as np

# Ma trận A
A = np.array([[3,2,2], [2, 1 ,-4], [6, 3 ,-2], [1, 3 ,-2], [3,1,3], [0, 1 ,-1], [3, 1 ,-2], [1, 0 ,-2]])

# Phân tích SVD
U, Sigma, VT = np.linalg.svd(A)

U = np.round(U,2)
Sigma = np.round(Sigma,2)
VT = np.round(VT,2)
# Kết quả
print("Ma trận U:")
print(U)
print("\nGiá trị kỳ dị Sigma:")
print(Sigma)
print("\nMa trận V^T:")
print(VT)
