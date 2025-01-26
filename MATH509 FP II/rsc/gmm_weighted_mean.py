import numpy as np
from sklearn.mixture import GaussianMixture

class GMMWeightedMean():
    def __init__(self, n_components=3, random_state=0):
        self.n_components = n_components
        self.random_state = random_state
        self.gmm = GaussianMixture(n_components=self.n_components, random_state=self.random_state)

    def fit(self, group, value):
        # 将特征和红颜色强度组合
        data_combined = np.hstack((group, value))
        # 拟合 GMM 模型
        self.gmm.fit(data_combined)

    def predict(self, new_data):
        # 初始化存储预测结果的列表
        predicted_intensity = []
        
        # 给新数据添加一列占位符
        new_data_with_placeholder = np.hstack((new_data, np.zeros((new_data.shape[0], 1))))
        
        # 对每个新数据点进行加权平均预测
        for point in new_data_with_placeholder:
            # 计算该点在每个成分的概率
            probabilities = self.gmm.predict_proba([point])[0]
            # 计算加权平均强度
            weighted_mean_intensity = np.dot(probabilities, self.gmm.means_[:, -1])
            predicted_intensity.append(weighted_mean_intensity)
        
        return np.array(predicted_intensity)


if __name__ == "__main__":
    ######## EXAMPLE #########

    group_red = np.random.rand(24, 2)  # 特征 (24, 2)
    how_red = np.random.rand(24, 1)    # 红颜色强度 (24, 1)

    # 创建 GMMWeightedMean 实例并拟合数据
    gmm_model = GMMWeightedMean(n_components=3, random_state=0)
    gmm_model.fit(group_red, how_red)

    # 生成新的 (10, 2) 数据并预测红颜色强度
    new_data = np.random.rand(10, 2)
    predicted_intensity = gmm_model.predict(new_data)

    print("新的数据:", new_data)
    print("加权平均预测的红颜色强度:", predicted_intensity)
