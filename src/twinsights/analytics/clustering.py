from ctypes import Union
from typing import List

import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from twinsights.analytics.comatrix import CoMatrix
from twinsights.analytics.data import AnalyticDataStore
from twinsights.analytics.visualization import generate_wordcloud
from twinsights.project import Project


def determine_k(comatrix: 'CoMatrix',
                normalize: bool = False,
                max_k=15) -> int:
    if normalize:
        df = preprocessing.normalize(comatrix.matrix)
    else:
        df = comatrix.matrix
    inertias = []
    for i in range(1, max_k):
        model = KMeans(n_clusters=i)
        model.fit(df)
        inertias.append(model.inertia_)
    from kneed import KneeLocator
    return KneeLocator(x=range(1, max_k),
                       y=inertias,
                       curve='convex',
                       direction='decreasing',
                       S=1).elbow


class Clustering:
    CLUSTER_COLUMN = '---CLUSTER---'

    def __init__(self,
                 project: Project,
                 name: str):
        self.clusters: Union[pd.DataFrame, None] = None
        self.project = project
        self.name = name
        self.k = 1

    def cluster(self,
                comatrix: CoMatrix,
                k: int,
                normalize: bool = False):
        if k <= 1:
            k = determine_k(comatrix, normalize)
        kmeans = KMeans(n_clusters=k, random_state=42)
        if normalize:
            normed = preprocessing.normalize(comatrix.matrix)
        else:
            normed = comatrix.matrix
        comatrix.matrix[Clustering.CLUSTER_COLUMN] = kmeans.fit_predict(normed)
        self.clusters = comatrix.matrix[Clustering.CLUSTER_COLUMN].copy()
        self.k = k

    def save(self,
             db):
        self.clusters.to_sql(self.name,
                             con=db.engine,
                             index=True,
                             if_exists="replace")

    @staticmethod
    def load(comatrix_name: str,
             project: Project,
             db: AnalyticDataStore) -> 'Clustering':
        cluster = Clustering(project, f"{comatrix_name}_clusters")
        cluster.clusters = pd.read_sql(f"SELECT * FROM {cluster.name}",
                                       con=db.engine)
        cluster.clusters.set_index('item1', inplace=True)
        cluster.k = len(pd.unique(cluster.clusters[Clustering.CLUSTER_COLUMN]))
        return cluster

    def visualize(self,
                  comatrix: CoMatrix):
        df = comatrix.matrix.merge(self.clusters, left_index=True,
                                   right_index=True)
        pca = PCA(n_components=2)
        principal_components = pd.DataFrame(
            pca.fit_transform(preprocessing.normalize(comatrix.matrix)),
            columns=['pca1', 'pca2'])
        index_name = df.index.name
        df.reset_index(inplace=True)
        df['pca1'] = principal_components['pca1']
        df['pca2'] = principal_components['pca2']
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        df.set_index(index_name, inplace=True)
        k = self.clusters[Clustering.CLUSTER_COLUMN].max()
        for cluster in range(k + 1):
            cdf = df[Clustering.CLUSTER_COLUMN] == cluster
            plt.scatter(df.loc[cdf, 'pca1'], df.loc[cdf, 'pca2'])
        ax.legend([f'Cluster {i}' for i in range(0, k+1)])
        ax.set_title(
            f'{self.name.replace("_clusters", "")} Clusters')
        plt.tight_layout()
        return fig

    def tag_cloud(self,
                  comatrices: List[CoMatrix]):
        num_clusters = self.k
        fig, ax = plt.subplots(nrows=num_clusters, ncols=len(comatrices))
        fig.set_figheight(num_clusters * 6)
        fig.set_figwidth(6 * len(comatrices))
        plt.style.use(['dark_background'])
        color_maps = ['Blues', 'Reds', 'Purples', 'Oranges', 'Greens', 'Greys',
                      'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                      'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
        for col, comatrix in enumerate(comatrices):
            comatrix.matrix = comatrix.matrix.merge(self.clusters,
                                                    left_index=True,
                                                    right_index=True)
            if len(comatrices) == 1:
                ax[0].set_title(comatrix.name + '\n', size='24', color='white')
            else:
                ax[0, col].set_title(comatrix.name + '\n', size='24',
                                     color='white')
            for i in range(num_clusters):
                c_i = comatrix.matrix[
                    comatrix.matrix[Clustering.CLUSTER_COLUMN] == i]
                c_i = c_i.T
                c_i.drop(Clustering.CLUSTER_COLUMN, inplace=True)
                c_i['SUM'] = c_i.sum(axis=1)
                c_i = c_i['SUM'].to_dict()
                if len(comatrices) == 1:
                    generate_wordcloud(c_i, axis_off=False, width=600,
                                       height=600, ax=ax[i],
                                       colormap=color_maps[i % len(color_maps)])
                else:
                    generate_wordcloud(c_i, axis_off=False, width=600,
                                       height=600, ax=ax[i, col],
                                       colormap=color_maps[i % len(color_maps)])

        if len(comatrices) == 1:
            for ax, row in zip(ax,
                               (f'Cluster {i}' for i in range(num_clusters))):
                ax.annotate(row, xy=(0, 0.5),
                            xytext=(-ax.yaxis.labelpad - 1, 0),
                            xycoords=ax.yaxis.label, textcoords='offset points',
                            rotation=0, size='24', ha='right', va='center',
                            color='white')
        else:
            for ax, row in zip(ax[:, 0],
                               (f'Cluster {i}' for i in range(num_clusters))):
                ax.annotate(row, xy=(0, 0.5),
                            xytext=(-ax.yaxis.labelpad - 1, 0),
                            xycoords=ax.yaxis.label, textcoords='offset points',
                            rotation=0, size='24', ha='right', va='center',
                            color='white')

        fig.align_labels()
        plt.tight_layout(pad=1)
        return fig
