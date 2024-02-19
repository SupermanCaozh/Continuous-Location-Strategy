import pandas as pd
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import os
from openpyxl import load_workbook

plt.rcParams["font.family"] = ['SimHei']  # 用来设定字体样式
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来设定无衬线字体样式


def plot_dendrogram(model, **kwargs):
    # 创建链接矩阵，然后绘制树状图

    # 在每个节点下对样本计数
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # 叶节点
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # 绘制树状图
    dendrogram(linkage_matrix, **kwargs)


def cal_dis(a, b):
    """
    计算两坐标点欧式距离
    :param a: [xa,ya]
    :param b: [xb,yb]
    :return:
    """
    distance = np.sqrt(np.sum(np.square(np.array(a) - np.array(b))))
    return distance


def compare_cord(x1, x2, eps):
    """
    比较两个坐标差异
    :param x1:
    :param x2:
    :return:
    """
    # eps = 0.001
    # if abs(x1[0] - x2[0]) > eps or abs(x1[1] - x2[1]) > eps:
    if abs(x1[0] - x2[0]) < eps and abs(x1[1] - x2[1]) < eps:
        # if x1[0] == x2[0] and x1[1] == x2[1]:
        return True
    else:
        return False


def stop_criteria(old, new, num_cluster):
    """
    计算精确重心法迭代判断条件,即两组重心差异未超过精度
    :param old:上一轮重心
    :param new:本轮重心
    :return:
    """
    # old = np.array(old)
    # new = np.array(new)
    # ds = np.array([cal_dis(old[i], new[i]) for i in range(len(old))])
    eps = 0.000001  # 绝对精度
    # if ds.any() > eps:
    #     return False
    # else:
    #     return True
    n_match = 0
    for n in new:
        for i in range(len(old)):
            match_flag = compare_cord(n, old[i], eps)
            if match_flag == True:
                n_match += 1
                old.remove(old[i])
                break
            else:
                continue
    if n_match == num_cluster:
        return True
    else:
        return False


def stop_criteria4iter(old, new, num_cluster):
    flag = 0
    eps = 0.0000001  # 绝对精度
    for i in range(num_cluster):
        if compare_cord(old[i], new[i], eps) == True:
            continue
        else:
            flag += 1
            return False
    if flag == 0:
        return True


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
    # plt.show()
    plt.savefig(os.path.join(r"gravity_fig", str(num_cluster) + " clusters.png"))

# 计算成本
def cost2(clusters, num):
    """
    计算经营成本和出行成本
    :param clusters: [gravity,region1,...,regionM]
    :return:
    """
    # 计算出行成本
    cost_trl = []
    # 计算经营成本
    cost_opr = []
    for i in range(num):
        cost1 = 0
        cost22 = 1500
        cost3 = 4  # 计算人事费
        pops = 0
        gravity = clusters[i][0]
        for cord in clusters[i][1:]:
            d = cal_dis(gravity, cord) * k
            idx = regions_cord.index(cord)
            pop = region_pop[idx]
            cost1 += pop * 0.12 * d

            pops += pop
        cost_trl.append(cost1)
        # 计算经营成本
        cost22 += max(0, 500 * (np.floor(pops / 100000) - 1))
        cost22 = cost22 * (22 + 4)

        cost3 += max(0, (np.floor(pops / 100000) - 1) * 1)
        cost3 = cost3 * 21000
        cost_opr.append(cost22 + cost3)
    return np.sum(cost_opr), cost_opr, np.sum(cost_trl), cost_trl


# 根据多选址策略中的重心法
# 先对克利夫兰地区的人口区域进行聚类
k = 2.5  # 度量因子,观察地图可得
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


# DBscan
# dbscan = DBSCAN(eps=1.414, min_samples=5)
# dbscan.fit(regions_cord)
# labels = dbscan.labels_

def main(num_cluster):
    # Kmeans,结果不稳定
    # num_cluster = 3
    # kmeans = KMeans(n_clusters=num_cluster)
    # kmeans.fit(regions_cord)
    # labels = kmeans.labels_
    # print(labels)

    # 层次聚类法
    model = AgglomerativeClustering(n_clusters=num_cluster)
    # model = AgglomerativeClustering(n_clusters=None,distance_threshold=0)
    model.fit(regions_cord)
    labels = model.labels_

    # fig=plt.figure()
    # plot_dendrogram(model, truncate_mode="level", p=3)
    # plt.title("层次聚类树状图")
    # plt.xlabel("簇中样本点个数")
    # plt.tight_layout()
    # plt.show()

    # 将人口区域按照聚类结果绘制

    # fig = plt.figure(dpi=200)
    # for cord in regions_cord:
    #     idx = regions_cord.index(cord)
    #     color = colors[labels[idx]]
    #     plt.scatter(cord[0], cord[1], c="xkcd:" + color)
    # plt.grid()
    # plt.show()

    # 1. 精确重心法解决多目标选址问题：
    # 初始化分类人口区域使用Kmeans方法,
    # 通过迭代不断修正分类人口区域并重新定位重心,直至连续2次重心位置无变化
    num_labels = num_cluster
    gravities = []
    rate = 0.12
    for i in range(num_cluster):
        labels_set = list(set(labels))
        label = labels_set[i]
        weights = 0
        g = [0, 0]
        for j in range(len(regions_cord)):
            if labels[j] != label:
                continue
            else:
                weight = region_pop[j] * rate
                weights += weight
                g[0] += weight * regions_cord[j][0]
                g[1] += weight * regions_cord[j][1]
        g[0] = g[0] / weights
        g[1] = g[1] / weights
        gravities.append(g)
    print("Initialization completed.")

    gravities = sorted(gravities, key=lambda x: (x[0] ** 2 + x[1] ** 2) ** (1 / 2))

    # 设置迭代停止条件
    iter_count = 0
    cost_transport = []  # 记录每一次迭代的克利夫兰地区出行总成本收敛过程
    while True:
        # 计算人口区域各坐标距离kmeans初始化所得重心
        clusters = [[gravities[i]] for i in range(num_cluster)]  # 每一个cluster[0]是重心位置,[1:-1]是群落其他点坐标
        for i in range(len(regions_cord)):
            ds = [cal_dis(regions_cord[i], gravities[j]) for j in range(num_cluster)]  # 计算当前region与其他各gravity之间的距离取最短下标
            d = min(ds)
            idx = ds.index(d)
            # 由于居民仅根据距离远近选择管理局故对其进行修正,重新聚类并确定重心
            clusters[idx].append(regions_cord[i])

        # 计算当前布局的居民出行总成本
        cost_trans = 0
        for i in range(num_cluster):
            for cord in clusters[i][1:]:
                d = cal_dis(gravities[i], cord) * k
                idx = regions_cord.index(cord)
                cost_trans += region_pop[idx] * rate * d
        cost_transport.append(cost_trans)

        # 根据最小化区域人口的出行成本计算重心
        temp_gravities = []
        for j in range(num_cluster):
            if len(clusters[j]) != 1:
                weights = 0
                g = [0, 0]
                for cord in clusters[j][1:]:
                    idx = regions_cord.index(cord)
                    weight = region_pop[idx] * rate
                    weights += weight
                    g[0] += weight * cord[0]
                    g[1] += weight * cord[1]
                g[0] = g[0] / weights
                g[1] = g[1] / weights
                temp_gravities.append(g)
            else:
                temp_gravities.append(clusters[j][0])
        # 修正过程
        for i in range(num_cluster):
            if len(clusters[i]) != 1:
                weights = 0
                g = [0, 0]
                for cord in clusters[i][1:]:
                    idx = regions_cord.index(cord)
                    d = cal_dis(cord, temp_gravities[i])  # 根据计算得出的重心,计算cluster内各点与其的距离,更新重心坐标
                    weight = region_pop[idx] * rate / d
                    weights += weight
                    g[0] += weight * cord[0]
                    g[1] += weight * cord[1]
                g[0] = g[0] / weights
                g[1] = g[1] / weights
                temp_gravities[i] = g
            else:
                continue

        temp_gravities = sorted(temp_gravities, key=lambda x: (x[0] ** 2 + x[1] ** 2) ** (1 / 2))

        # stop = stop_criteria(gravities, temp_gravities, num_cluster)
        stop = stop_criteria4iter(gravities, temp_gravities, num_cluster)
        if stop == True:
            iter_count += 1
            print("Iteration {} completed".format(iter_count))
            plot_gravity(clusters, num_cluster)

            break
        else:
            gravities = temp_gravities
            iter_count += 1
            print("Iteration {} completed".format(iter_count))
            # plot_gravity(clusters, num_cluster)
            continue
    # 当迭代终止时,temp_gravities为最优布局
    for i in range(len(temp_gravities)):
        print(temp_gravities[i])

    # 根据最终迭代所得重心确定群落分配
    clusters = [[temp_gravities[i]] for i in range(num_cluster)]  # 每一个cluster[0]是重心位置,[1:-1]是群落其他点坐标
    for i in range(len(regions_cord)):
        ds = [cal_dis(regions_cord[i], temp_gravities[j]) for j in
              range(num_cluster)]  # 计算当前region与其他各gravity之间的距离取最短下标
        d = min(ds)
        idx = ds.index(d)
        # 由于居民仅根据距离远近选择管理局故对其进行修正,重新聚类并确定重心
        clusters[idx].append(regions_cord[i])

    cost_opr, cost_opr_each, cost_trl, cost_trl_each = cost2(clusters, num_cluster)

    print("The optimal cost is {}".format(cost_opr + cost_trl))

    fig = plt.figure(dpi=200)
    plt.plot(cost_transport, marker='.', c="red", linewidth=1.5)
    plt.xlabel("Number of Iteration")
    plt.ylabel("Cost of Travelling")
    plt.grid(ls="--")
    plt.show()

    return temp_gravities, cost_opr, cost_opr_each, np.mean(cost_opr_each), cost_trl, cost_trl_each, np.mean(
        cost_trl_each)


if __name__ == "__main__":
    data = pd.DataFrame(columns=["best layout", "cost_opr", "cost_opr_each", "cost_opr_ave",
                                 "cost_trl", "cost_trl_each", "cost_trl_ave"], index=np.arange(1, 11))
    for i in range(1, 11):
        num_cluster = i
        gravities, cost_opr, cost_opr_each, cost_opr_ave, cost_trl, cost_trl_each, cost_trl_ave = main(num_cluster)
        data.loc[i, "best layout"] = gravities
        data.loc[i, "cost_opr"] = cost_opr
        data.loc[i, "cost_opr_each"] = cost_opr_each
        data.loc[i, "cost_opr_ave"] = cost_opr_ave
        data.loc[i, "cost_trl"] = cost_trl
        data.loc[i, "cost_trl_each"] = cost_trl_each
        data.loc[i, "cost_trl_ave"] = cost_trl_ave

    # data.to_excel(r"data_gravity.xlsx", sheet_name="层次聚类指定聚类数", header=True, index=True)
