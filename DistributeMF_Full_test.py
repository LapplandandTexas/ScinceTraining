import time
import copy
import numpy as np

from load_data_test import ratings_dict, ratings_matrix
from shared_parameter import *

def user_update(user_matrix, item_matrix, rating_matrix, m, n):
    # 初始化学习率
    reg = 0.01
    print("原矩阵")
    print(rating_matrix)
    for epoch in range(10):
        # 固定item更新user
        for i in range(m):
            user_matrix[i] = np.linalg.solve(reg * np.eye(5) + np.dot(item_matrix, item_matrix.T), np.dot(rating_matrix[i], item_matrix.T))
        # 固定user更新item
        for j in range(n):
            item_matrix[:, j] = np.linalg.solve(reg * np.eye(k) + np.dot(user_matrix.T, user_matrix), np.dot(rating_matrix[:, j], user_matrix))
    return user_matrix, item_matrix

def loss(rating, user, item, m, n):
    loss = []
    for i in range(m):
        for j in range(n):
            error = (rating[i][j] - np.dot(user[i], item[:,j])) ** 2
            loss.append(error)
    return np.mean(loss)

if __name__ == '__main__':

    m, n = ratings_matrix.shape
    user_matrix_full = np.random.normal(0, 0, (m, 5))
    start_place = 0

    print(user_matrix_full)

    for i in range(10):
        # print(ratings_dict)
        ratings = ratings_dict[i + 1]
        m, n = ratings.shape
        k = 5
        user_matrix = np.random.normal(0, 0.1, (m, k))
        item_matrix = np.random.normal(0, 0.1, (k, n))
        user_matrix, item_matrix = user_update(user_matrix, item_matrix, ratings, m, n)
        print("预测矩阵")
        print(np.dot(user_matrix, item_matrix))
        user_matrix_full[start_place:start_place + m] = user_matrix[0:m]
        # print("查看矩阵数据替换问题")
        # print(user_matrix_full[start_place:start_place + m])
        # print(user_matrix[0:m])
        start_place += m
        print("loss:")
        print(loss(ratings, user_matrix, item_matrix, m, n))

    print("整个user")
    print(user_matrix_full)





