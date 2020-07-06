from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import seaborn as sns


def plot_clusters_2d(X_df,y_labels,pc):

    kmeans_model = KMeans(n_clusters=7, init='k-means++', random_state=101)
    y_cluster = kmeans_model.fit(pc)
    # y_cluster_cent = kmeans_model.fit_predict(pccent)

    colors = ['#e74c3c', 'dodgerblue', 'purple', '#2ecc71', '#ff33cc', 'orange', '#9b59b6']
    ax2 = sns.scatterplot(x=pc[:, 0], y=pc[:, 1], hue=y_cluster.labels_, palette=colors)
    ax2.set(xlabel="pc1", ylabel="pc2", title="Credit card users clustering result")
    ax2.legend(title='cluster')
    ax2 = sns.scatterplot(y_cluster.cluster_centers_[:, 0], y_cluster.cluster_centers_[:, 1],
                          hue=range(7), palette=colors, s=25, ec='black',marker="D", legend=False, ax=ax2)

    # ax2.scatter(pccent[:,0], pccent[:,1], c='black')

    plt.show()


def getLabelsKMeans(pc):
    kmeans_model = KMeans(n_clusters=7, init='k-means++', random_state=101)
    y_cluster = kmeans_model.fit(pc)
    # y_cluster_cent = kmeans_model.fit_predict(pccent)

    return y_cluster.labels_


def getElbow(data):
    sum_of_squared_distance = []
    n_cluster = range(1, 20)

    for k in n_cluster:
        kmean_model = KMeans(n_clusters=k)
        kmean_model.fit(data)
        sum_of_squared_distance.append(kmean_model.inertia_)

    plt.plot(n_cluster, sum_of_squared_distance, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum of squared distances')
    plt.title('Elbow method for optimal K')
    plt.show()



