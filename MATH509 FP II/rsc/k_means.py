import numpy as np
from sklearn.cluster import KMeans

class KMeansWeightedMean():
    def __init__(self, n_clusters=3, random_state=0):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)

    def fit(self, group, value):
        # 将特征和红颜色强度组合
        data_combined = np.hstack((group, value))  # shape (24, 3)
        # 拟合 KMeans 模型
        self.kmeans.fit(data_combined)

    def predict(self, new_data):
        # 确保新的数据与训练时数据的特征数一致，给新数据添加占位符
        new_data_with_placeholder = np.hstack((new_data, np.zeros((new_data.shape[0], 1))))  # shape (10, 3)
        
        # 初始化存储预测结果的列表
        predicted_intensity = []

        # 对每个新数据点进行加权平均预测
        for point in new_data_with_placeholder:
            # 获取该点所属簇的索引
            cluster_index = self.kmeans.predict([point])[0]
            # 获取该簇中心的红颜色强度
            cluster_center = self.kmeans.cluster_centers_[cluster_index, -1]
            predicted_intensity.append(cluster_center)
        
        return np.array(predicted_intensity)


if __name__ == "__main__":
    ######## EXAMPLE #########

    group_red = np.random.rand(24, 2)  # 特征 (24, 2)
    how_red = np.random.rand(24, 1)    # 红颜色强度 (24, 1)

    # 创建 KMeansWeightedMean 实例并拟合数据
    kmeans_model = KMeansWeightedMean(n_clusters=3, random_state=0)
    kmeans_model.fit(group_red, how_red)

    # 生成新的 (10, 2) 数据并预测红颜色强度
    new_data = np.random.rand(30, 2)
    predicted_intensity = kmeans_model.predict(new_data)

    print("新的数据:", new_data)
    print("加权平均预测的红颜色强度:", predicted_intensity)
