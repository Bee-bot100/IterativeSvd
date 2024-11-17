#from iterative_SVD import iterative_svd
import numpy as np 
import pandas as pd
from scipy.sparse.linalg import svds

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

file = pd.read_excel("C:\\Users\\ADMIN\\OneDrive\\Desktop\\Codespace\\Python\\matrix.xlsx")


# Giả sử có ma trận đánh giá với các giá trị thiếu
matrix = (file.values[:, 1:])

#Lấy giá trị trung bình của các hàng
mean_user_rating = np.nanmean(matrix, axis=1)

K = {}

#Thay thế các giá trị nan bằng trung bình hàng của nó
for i in range(matrix.shape[0]):
    Q = []
    for k in range(matrix.shape[1]):
        if np.isnan(matrix[i][k]):
            matrix[i][k] = mean_user_rating[i]
            Q.append(k)
    if len(Q) != 0:
        K[f"{i}"] = Q
# Chuẩn hóa ma trận
rating_matrix = matrix - mean_user_rating.reshape(-1, 1)
rating_matrix = rating_matrix.astype(float)

#Phân tích iterative svd cho ma trận trên
Q,Z,Pt = svds(rating_matrix, k = 2)
Z = Z_(Q,Z,np.transpose(Pt))

# Tái tạo ma trận xấp xỉ
predicted_ratings = np.dot(np.dot(Q, Z), Pt) + mean_user_rating.reshape(-1, 1)
predicted_ratings = np.round(predicted_ratings.astype(float),1)

#print(predicted_ratings)

#Check
inp = int(input("Nguoi dung so: "))
row = inp-1
selectrow = file.iloc[[inp-1]]
if str(row) in K:
    print(selectrow)
    print("Dự đoán: ",end="")
    for i in K[str(row)]:
        print(predicted_ratings[row][i],end = "  ")
else:
    print("Rated:")
    print(selectrow)
