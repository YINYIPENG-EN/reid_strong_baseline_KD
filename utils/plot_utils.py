import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


# def feat_plot(n_clusters, feat):
#
#     kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(feat)
#     # 获取每个样本的簇标签
#     labels = kmeans.labels_
#     # 将输出张量降维到二维空间中
#     tsne = TSNE(n_components=2, init='random', random_state=0)
#     output_tsne = tsne.fit_transform(feat)
#
#     # 将聚类结果可视化
#     colors = plt.cm.rainbow(np.linspace(0, 1, n_clusters))
#     for i in range(n_clusters):
#         plt.scatter(output_tsne[labels == i, 0], output_tsne[labels == i, 1], s=10, color=colors[i])
#     plt.show()
#     #plt.savefig('./logs/feat.png')
#     plt.close()

feat_list = []
for i in range(100):
    feat = torch.randn(8, 10)
    X = torch.cat([feat], dim=1).numpy()
    feat_list.append(X)
feat_arry = np.array(feat_list)
n_samples, _, n_features = feat_arry.shape
print("n_samples: ", n_samples)
feat_arry = feat_arry.reshape((n_samples, -1))
kmean = KMeans(n_clusters=10)
labels = kmean.fit_predict(feat_arry)

# 将特征降到2维，用于可视化
pca = PCA(n_components=2)
reduced_feats = pca.fit_transform(feat_arry)

# 可视化聚类结果
plt.scatter(reduced_feats[:, 0], reduced_feats[:, 1], c=labels)
plt.show()