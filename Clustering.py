import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

Dataset = pd.read_csv('wine-clustering.csv')
Dataset.info()
Data_np = Dataset.to_numpy()
scaler = StandardScaler()
scaler.fit(Data_np)
Data_st = scaler.transform(Data_np)
# Обработка данных методом главных компонент и их визуализация
pca = PCA(n_components=2)
pca.fit(Data_st)
Data_pca = pca.fit_transform(Data_st)
plt.xlim(Data_pca[:, 0].min(), Data_pca[:, 0].max()+1)
plt.ylim(Data_pca[:, 1].min(), Data_pca[:, 1].max()+1)
for i in range(len(Data_np)):
    plt.plot(Data_pca[i, 0], Data_pca[i, 1], marker='.')
plt.show()
# Обработка данных методом t-SNE и их визуализация
tsne = TSNE()
Data_tsne = tsne.fit_transform(Data_st)
plt.xlim(Data_tsne[:, 0].min(), Data_tsne[:, 0].max()+1)
plt.ylim(Data_tsne[:, 1].min(), Data_tsne[:, 1].max()+1)
for i in range(len(Data_np)):
    plt.plot(Data_tsne[i, 0], Data_tsne[i, 1], marker='.')
plt.show()
print('Из графиков видно, что количество классов для кластеризации: 3')
# Кластеризация методом "К средних"
kmeans = KMeans(n_clusters=3)
kmeans.fit(Data_pca)
colors = ['#FF0000', '#008000', '#0000FF']
plt.xlim(Data_pca[:, 0].min(), Data_pca[:, 0].max()+1)
plt.ylim(Data_pca[:, 1].min(), Data_pca[:, 1].max()+1)
for i in range(len(Data_np)):
    plt.text(Data_pca[i, 0], Data_pca[i, 1],
             str(kmeans.labels_[i]),
             color=colors[kmeans.labels_[i]],
             fontdict={'weight': 'bold', 'size': 9})
plt.show()
