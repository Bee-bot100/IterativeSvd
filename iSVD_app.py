from iterative_SVD import iterative_svd
import numpy as np 
import pandas as pd

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

Q,Z,P = iterative_svd(rating_matrix)

# Tái tạo ma trận xấp xỉ
predicted_ratings = np.dot(np.dot(Q, Z), np.transpose(P)) + mean_user_rating.reshape(-1, 1)
predicted_ratings = np.round(predicted_ratings.astype(float),2)

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
