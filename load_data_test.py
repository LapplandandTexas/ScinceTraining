import os
import csv

import numpy as np
from surprise import AlgoBase
np.set_printoptions(precision=2, suppress=True)

def load_csv(fileName, fileWithHeader=True):
    with open(fileName, 'r' ,encoding='utf-8') as f:
        lines = f.readlines()
        if fileWithHeader:
            header = lines[0].strip().split(',')
            data = [line.strip().split(',') for line in lines[1:]]
        else:
            header = []
            data = [line.strip().split(',') for line in lines]
    return header, data


predict_step = 3
least_rating_num = 5

current_path = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(current_path, 'ml-latest-small')

headers, ratings = load_csv(os.path.join(data_path, 'ratings.csv'))

headers, movies = load_csv(os.path.join(data_path, 'movies.csv'))

# 生成用户list
user_id_list =[]
for e in ratings:
    id = int(e[0])
    if id in user_id_list:
        continue
    else:
        user_id_list.append(id)
# print(user_id_list)

# 生成项目list
item_id_list = [int(e[0]) for e in movies]

# print(len(user_id_list))
# print(len(item_id_list))


# 将用户list(m*1)与项目list(1*n)转换为矩阵形式
user_id_matrix = np.array(user_id_list).reshape(-1, 1)
user_id_matrix = np.insert(user_id_matrix, 0, 0, axis=0)
item_id_matrix = np.array(item_id_list).reshape(1, len(item_id_list))
item_id_matrix = np.insert(item_id_matrix, 0, 0, axis=1)
# 初始化评分矩阵
ratings_matrix = np.zeros((user_id_matrix.shape[0], item_id_matrix.shape[1]), dtype=float)
ratings_matrix[:, 0] = user_id_matrix[:, 0]
ratings_matrix[0, :] = item_id_matrix[0, :]
index_row = ratings_matrix[1:, 0]
index_col = ratings_matrix[0, 1:]
# 导入评分数值
for e in ratings:
    change_row = np.where(index_row == int(e[0]))[0][0] + 1
    change_col = np.where(index_col == int(e[1]))[0][0] + 1
    ratings_matrix[change_row, change_col] = float(e[2])
ratings_matrix = np.delete(ratings_matrix, 0, axis=0)
# ratings_matrix = np.delete(ratings_matrix, 0, axis=1)
# print(ratings_matrix)

# 数据分割
# 打乱评分矩阵
np.random.shuffle(ratings_matrix)
# print(ratings_matrix)
ratings_matrix = np.delete(ratings_matrix, 0 ,axis=1)
# 对打乱的矩阵进行分割
ratings_dict = {}
for i in range(1,11):
    part_matrix = ratings_matrix[61 * (i - 1):61 * i]
    ratings_dict[i] = part_matrix
# print(ratings_dict)
# print(ratings_dict[10])










# # ALS算法实现矩阵分解
# m = len(user_id_list)
# n = len(item_id_list)
# k = 5
# reg = 0.01

# # 初始化用户隐矩阵和项目隐矩阵
# user_matrix = np.random.normal(0, 0.1, (m, k))
# item_matrix = np.random.normal(0, 0.1, (k, n))
# # 标记缺损值
# mask = np.where(ratings_matrix != 0, 1, 0)
# for epoch in range(20):
#     # 固定item更新user
#     for i in range(m):
#         mask_i = mask[i, :]
#         ratings_i = ratings_matrix[i, mask_i]
#         item_matrix_i = item_matrix[:, mask_i]
#         user_matrix[i] = np.linalg.solve(reg * np.eye(k) + np.dot(item_matrix_i, item_matrix_i.T), np.dot(ratings_i, item_matrix_i.T))
#     # 固定user更新item
#     for j in range(n):
#         mask_j = mask[:, j]
#         ratings_j = ratings_matrix[mask_j, j]
#         user_matrix_j = user_matrix[mask_j, :]
#         item_matrix[:, j] = np.linalg.solve(reg * np.eye(k) + np.dot(user_matrix_j.T, user_matrix_j), np.dot(ratings_j, user_matrix_j))
# print(np.dot(user_matrix, item_matrix) * mask)

# def fit(self, trainset):
#     AlgoBase.fit(self, trainset)
#
#     # 初始化用户和物品因子矩阵
#     pu = np.random.normal(0, 0.1, (self.trainset.n_users, self.n_factors))
#     qi = np.random.normal(0, 0.1, (self.trainset.n_items, self.n_factors))
#
#     # 训练模型
#     for epoch in range(self.n_epochs):
#         for u, i, r in self.trainset.all_ratings():
#             # 计算误差
#             e_ui = r - np.dot(pu[u], qi[i])
#
#             # 更新用户和物品因子矩阵
#             pu[u] += self.reg_pu * (e_ui * qi[i] - self.reg_pu * pu[u])
#             qi[i] += self.reg_qi * (e_ui * pu[u] - self.reg_qi * qi[i])
#
#     self.pu = pu
#     self.qi = qi
