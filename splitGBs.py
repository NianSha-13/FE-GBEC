import numpy as np
from gbutils.granular_ball import GranularBall
from gbutils.HyperballClustering import *
# from functions.wkmeans_no_random import WKMeans
# from sklearn.cluster import k_means


def splitGBs(data,all_indices):
    gb_list_temp = [data]  # 粒球集合[ [[data1],[data2],...], [[data1],[data2],...],... ],初始只有一个粒球
    division_num = 0  # 记录第几次分裂
    # 当粒球不再分裂停止

    index=[all_indices]
    while 1:
        ball_number_old = len(gb_list_temp)
        gb_list_temp,index = division_2_2(gb_list_temp,index)  # 粒球划分
        ball_number_new = len(gb_list_temp)
        if ball_number_new == ball_number_old:
            break

    radius = []  # 汇总所有粒球半径
    for gb_data in gb_list_temp:
        if len(gb_data) >= 2:  # 粒球中样本多于2个时，认定为合法粒球，并收集其粒球半径
            radius.append(get_radius(gb_data[:, :]))
    radius_median = np.median(radius)  #半径的中位数
    radius_mean = np.mean(radius)  #半径的均值
    # 中位数与均值的最小值
    radius_detect = min(radius_median, radius_mean)

    # 缩小粒球半径 将粒球半径大于radius_detect的进行缩小
    while 1:
        ball_number_old = len(gb_list_temp)
        gb_list_temp,index = minimum_ball(gb_list_temp, radius_detect,index)  # 缩小粒球 将粒球半径大于radius_detect的进行缩小
        ball_number_new = len(gb_list_temp)
        if ball_number_new == ball_number_old:
            break

    gb_list = []
    for obj in gb_list_temp:
        # 去噪声，移除1个点的球
        # if len(obj) == 1:
        #     continue
        gb = GranularBall(obj,index)
        gb_list.append(gb)
    return gb_list,index


# def wsplitGBs(data):
#     hb_list_temp = [data]
#     hb_list_not_temp = []
#     division_num = 0
#     # 按照dm值分裂
#     while 1:
#         ball_number_old = len(hb_list_temp) + len(hb_list_not_temp)
#         division_num = division_num + 1
#         hb_list_temp, hb_list_not_temp = division(hb_list_temp, hb_list_not_temp, division_num, K=2)  # 加权分裂，每次一分为2
#         ball_number_new = len(hb_list_temp) + len(hb_list_not_temp)
#         if ball_number_new == ball_number_old:
#             hb_list_temp = hb_list_not_temp
#             break
#
#     # radius = []  # 汇总所有粒球半径
#     # for gb_data in hb_list_temp:
#     #     if len(gb_data) >= 2:  # 粒球中样本多于2个时，认定为合法粒球，并收集其粒球半径
#     #         radius.append(get_radius(gb_data[:-1, :]))
#     # radius_median = np.median(radius)
#     # radius_mean = np.mean(radius)
#     # radius_detect = min(radius_median, radius_mean)
#     #
#     # while 1:
#     #     ball_number_old = len(hb_list_temp)
#     #     hb_list_temp = minimum_ball(hb_list_temp, radius_detect)  # 缩小粒球
#     #     ball_number_new = len(hb_list_temp)
#     #     if ball_number_new == ball_number_old:
#     #         break
#
#     dic_w = {}  # 每个球对应的维度的权重
#     ball_list = []
#     w_total = np.zeros((1, data.shape[1]))
#     for index, hb_all in enumerate(hb_list_temp):  # hb_all代表一个球，包含数据和权重
#         w = hb_all[-1, :]  # 取出每个球权重
#         hb_no_w = np.delete(hb_all, -1, axis=0)  # 取出每个球的数据
#         ball_list.append(hb_no_w)
#         dic_w[index] = w
#         w_total = w_total + w
#     ave_w = w_total / len(hb_list_temp)  # 平均维度权重
#
#     # 初始成粒球
#     gb_list = []
#     for i in range(0, len(ball_list)):
#         gb = GranularBall(ball_list[i], ave_w)
#         gb_list.append(gb)
#     return gb_list
#
#
# # 计算子球的质量,hb为粒球中所有的点
# def get_dm(hb, w):
#     num = len(hb)
#     center = hb.mean(0)
#     diff_mat = center - hb
#     w_mat = np.tile(w, (num, 1))
#     # 将w形状化为与diff_mat一致，然后做内积
#     sq_diff_mat = diff_mat ** 2 * w_mat
#     sq_distances = sq_diff_mat.sum(axis=1)
#     distances = sq_distances ** 0.5
#     sum_radius = 0
#     sum_radius = sum(distances)
#     if num > 1:
#         return sum_radius / num
#     else:
#         return 1
#
# def get_radius(gb_data):
#     # origin get_radius 7*O(n)
#     sample_num = len(gb_data)
#     center = gb_data.mean(0)
#     diffMat = np.tile(center, (sample_num, 1)) - gb_data
#     sqDiffMat = diffMat ** 2
#     sqDistances = sqDiffMat.sum(axis=1)
#     distances = sqDistances ** 0.5
#     radius = max(distances)
#     return radius
#
# def division(hb_list, hb_list_not, division_num, K):
#     gb_list_new = []
#     i = 0
#     split_threshold = K  # 控制粒球里至少包含的点的数量
#     for hb in hb_list:
#         hb_no_w = hb
#         # 如果不是第一次分裂
#         if division_num != 1:
#             parent_w = np.array([hb[-1][:]])  # 取出父球权重
#             hb_no_w = np.delete(hb, -1, axis=0)  # 取出父球数据
#             K = 2
#             split_threshold = 2
#         # 如果是第一次分裂
#         else:
#             n, m = hb.shape
#             parent_w = np.ones((1, m))
#         if len(hb_no_w) > split_threshold:
#             i = i + 1
#             if division_num != 1:
#                 K = 2  # 除了第一次分裂，后面一分为2
#             ball, child_w = spilt_ball_by_k(hb_no_w, parent_w, K, division_num)  # ball：所有子球，child_w： 子球每个维度的权重
#             flag = 0
#             for i in range(len(ball)):
#                 if len(ball[i]) == 0:
#                     flag = 1
#                     break
#             # 分裂成功
#             if flag == 0:
#                 # 子球的dm
#                 dm_child_ball = []
#                 # 子球的大小
#                 child_ball_length = []
#                 dm_child_divide_len = []
#                 for i in range(K):
#                     temp_dm = get_dm(ball[i], child_w)
#                     temp_len = len(ball[i])
#                     dm_child_ball.append(temp_dm)
#                     child_ball_length.append(temp_len)
#                     dm_child_divide_len.append(temp_dm * temp_len)
#                 w0 = np.array(child_ball_length).sum()
#                 dm_child = np.array(dm_child_divide_len).sum() / w0  # 子球加权dm
#                 dm_parent = get_dm(hb_no_w, parent_w)  # 父球dm
#                 t2 = (dm_child < dm_parent)
#                 if t2:
#                     # child_w = child_w.flatten()
#                     for i in range(K):
#                         # tt_ball = ball[i]
#                         temp_ball = np.append(ball[i], child_w, axis=0)  # 每个球最后一行不是数据，而是权重
#                         gb_list_new.extend([temp_ball])
#                 else:
#                     hb_list_not.append(hb)
#             # 分裂失败
#             else:
#                 hb_list_not.append(hb)
#         else:
#             hb_list_not.append(hb)
#
#     return gb_list_new, hb_list_not
#
#
# def spilt_ball_by_k(data_no_w, w, k, division_num):
#     centers = []  # 分裂中心
#     max_iter = 1
#     data = data_no_w
#     if division_num != 1 or k == 2:
#         k = 2  # 如果不是第一次分裂或者k=2，就一分为2
#         center = data.mean(0)
#         p_max1 = np.argmax(((data - center) ** 2).sum(axis=1) ** 0.5)
#         p_max2 = np.argmax(((data - data[p_max1]) ** 2).sum(axis=1) ** 0.5)
#         c1 = (data[p_max1] + center) / 2
#         c2 = (data[p_max2] + center) / 2
#         # 有初始质心
#         centers.append(c1)
#         centers.append(c2)
#     else:
#         centers = k_means(data, k, init="k-means++", n_init=10, random_state=42)[0]
#         max_iter = 10
#     model = WKMeans(n_clusters=k, max_iter=max_iter, belta=10, centers=centers, w=w)
#     cluster = model.fit_predict(data)
#     w = model.w
#     ball = []
#     for i in range(k):
#         ball.append(data_no_w[cluster == i, :])
#     return ball, w
#
#
# # 缩小粒球
# def minimum_ball(gb_list, radius_detect):
#     gb_list_temp = []
#     K = 2  # 一分为2
#     for gb_data in gb_list:
#
#         parent_w = gb_data[-1, :]  # 取出每个球权重
#         gb_data_no_w = np.delete(gb_data, -1, axis=0)  # 取出每个球的数据
#         # if len(hb) < 2: stream
#         if len(gb_data_no_w) <= 2:
#             # gb_list_temp.append(gb_data)
#
#             if (len(gb_data_no_w) == 2) and (get_radius(gb_data_no_w) > 1.2 * radius_detect):
#                 # print(get_radius(gb_data))
#                 gb_list_temp.append(np.array([gb_data[0], ]))
#                 gb_list_temp.append(np.array([gb_data[1], ]))
#
#             else:
#                 gb_list_temp.append(gb_data)
#         else:
#             # if get_radius(gb_data) <= radius_detect:
#             if get_radius(gb_data_no_w) <= 1.2 * radius_detect:
#                 gb_list_temp.append(gb_data)
#             else:
#                 # ball_1, ball_2 = spilt_ball(gb_data)  # 无模糊
#                 ball, child_w = spilt_ball_by_k(gb_data_no_w, parent_w, K, division_num=2)  # ball：所有子球，child_w： 子球每个维度的权重
#                 temp_ball = []
#                 for i in range(K):
#                     temp_ball = np.append(ball[i], child_w, axis=0)  # 每个球最后一行不是数据，而是权重
#                     # gb_list_new.extend([temp_ball])
#                 if len(temp_ball[0]) == 1 or len(temp_ball[1]) == 1:
#                     if get_radius(gb_data_no_w) > radius_detect:
#                         gb_list_temp.extend([temp_ball])
#                     else:
#                         gb_list_temp.append(gb_data)
#                 else:
#                     gb_list_temp.extend([temp_ball])
#
#     return gb_list_temp
#
#
# def spilt_ball(data):
#     ball1 = []
#     ball2 = []
#     n, m = data.shape
#     X = data.T
#     G = np.dot(X.T, X)
#     H = np.tile(np.diag(G), (n, 1))
#     D = np.sqrt(np.abs(H + H.T - G * 2))  # 计算欧式距离
#     r, c = np.where(D == np.max(D))  # 查找距离最远的两个点坐标
#     r1 = r[1]
#     c1 = c[1]
#     for j in range(0, len(data)):  # 对于距离最远的两个点之外的点，根据到两点距离不同划分到不同的簇中
#         if D[j, r1] < D[j, c1]:
#             ball1.extend([data[j, :]])
#         else:
#             ball2.extend([data[j, :]])
#     ball1 = np.array(ball1)
#     ball2 = np.array(ball2)
#     return [ball1, ball2]
