import pandas as pd
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import os
from openpyxl import load_workbook

plt.rcParams["font.family"] = ['SimHei']  # 用来设定字体样式
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来设定无衬线字体样式


def cal_dis(a, b):
    """
    计算两坐标点欧式距离
    :param a: [xa,ya]
    :param b: [xb,yb]
    :return:
    """
    distance = np.sqrt(np.sum(np.square(np.array(a) - np.array(b))))
    return distance


def plot_gravity(clusters, num_cluster):
    # 将最后的重心法结果绘制在地图上
    fig = plt.figure(dpi=200)
    for i in range(num_cluster):
        plt.scatter(clusters[i][0][0], clusters[i][0][1], marker="^", c="xkcd:" + colors[i], facecolor="none")
        regions = np.array([cord for cord in clusters[i][1:]])
        plt.scatter(regions[:, 0], regions[:, 1], marker=".", c="xkcd:" + colors[i])
        for cord in clusters[i][1:]:
            plt.plot([cord[0], clusters[i][0][0]], [cord[1], clusters[i][0][1]], c="xkcd:" + colors[i], ls="--",
                     linewidth=1)
    plt.grid(ls="--")
    plt.xlabel("网格列数")
    plt.ylabel("网格行数")
    plt.tight_layout()
    plt.show()
    # plt.savefig(os.path.join(r"gravity_fig", "present_nodes.png"))


# num_cluster = 8
num_cluster=2
k = 2.5
rate = 0.12
region_pop = [4100, 7800, 8100, 10700, 11500, 9300, 10100, 8800, 5300, 5100, 7700, 4300,
              6200, 8700, 10500, 12800, 13900, 14900, 12600, 13700, 16700, 17400, 9200, 6700,
              7200, 9400, 15600, 13800, 14500, 13700, 16700, 15200, 13800, 10300, 7500, 5800,
              10300, 11800, 10500, 15600, 13700, 10200, 15800, 14100, 11900, 9800, 8500, 6800,
              200, 100, 200, 400, 600, 1200, 12400, 10800, 13500, 10300, 7800, 5400,
              2600, 17200, 18600, 15500, 9900, 7100,
              500, 12000, 11700, 8700, 6400]
colors = ["purple", "green", "blue", "pink", "brown", "red",
          "light blue", "teal", "orange",
          "light green", "magenta", "yellow"]


# nodes = [[5.2, 3.0], [7.8, 5.5], [1.2, 2.5], [2.7, 1.3],
#          [5.9, 1.5], [4.1, 4.4], [9.0, 6.9], [11.2, 5.5]]

# gravity方法开设2个网点的具体位置
# nodes=[[4.083681851532504, 2.6035897017066914], [9.288274284050223, 4.072837920120164]]

# meta方法开设2个网点时的服务网络
nodes=[[4.51410865, 2.21016663],[9.12340151, 4.51481306]]
clusters=[[[4.51410865, 2.21016663], [1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [6, 1], [7, 1], [8, 1], [1, 2], [2, 2], [3, 2], [4, 2], [5, 2], [6, 2], [7, 2], [1, 3], [2, 3], [3, 3], [4, 3], [5, 3], [6, 3], [1, 4], [2, 4], [3, 4], [4, 4], [5, 4], [6, 4], [1, 5], [2, 5], [3, 5], [4, 5], [5, 5]],
       [[9.12340151, 4.51481306], [9, 1], [10, 1], [11, 1], [12, 1], [8, 2], [9, 2], [10, 2], [11, 2], [12, 2], [7, 3], [8, 3], [9, 3], [10, 3], [11, 3], [12, 3], [7, 4], [8, 4], [9, 4], [10, 4], [11, 4], [12, 4], [6, 5], [7, 5], [8, 5], [9, 5], [10, 5], [11, 5], [12, 5], [7, 6], [8, 6], [9, 6], [10, 6], [11, 6], [12, 6], [8, 7], [9, 7], [10, 7], [11, 7], [12, 7]]]

x_col = np.arange(1, 13)
y_row = np.arange(1, 8)  # 在该问题表示中,图中左下为(0.5,0.5)
regions_cord = []
for i in y_row[:-2]:
    for j in x_col:
        regions_cord.append([j, i])  # [col,row]
for i in x_col[6:]:
    regions_cord.append([i, 6])
for i in x_col[7:]:
    regions_cord.append([i, 7])

# 针对当前管理局网点布局分配服务区域

# clusters = [[nodes[i]] for i in range(num_cluster)]  # 每一个cluster[0]是重心位置,[1:-1]是群落其他点坐标
# for i in range(len(regions_cord)):
#     ds = [cal_dis(regions_cord[i], nodes[j]) for j in range(num_cluster)]  # 计算当前region与其他各gravity之间的距离取最短下标
#     d = min(ds)
#     idx = ds.index(d)
#     # 由于居民仅根据距离远近选择管理局故对其进行修正,重新聚类并确定重心
#     clusters[idx].append(regions_cord[i])
# plot_gravity(clusters, num_cluster)  # 画出当前网点布局图

# 计算当下的各服务区域居民出行成本
cost_trans = 0
cost_transport = []
pops_served = []
for i in range(num_cluster):
    single_cost = 0
    pops = 0
    for cord in clusters[i][1:]:
        d = cal_dis(nodes[i], cord) * k
        idx = regions_cord.index(cord)
        cost_trans += region_pop[idx] * rate * d
        single_cost += region_pop[idx] * rate * d
        pops += region_pop[idx]
    pops_served.append(pops)
    cost_transport.append(single_cost)

# 打印当前网点布局成本
print("总出行成本为{}".format(cost_trans))
print("各服务区域出行成本为{}".format(cost_transport))
print("各管理局服务人口数为{}".format(pops_served))
print(f"人口总数为{sum(region_pop), sum(pops_served)}")
