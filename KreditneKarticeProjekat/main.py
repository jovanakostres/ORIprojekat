import pandas
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial.distance import cdist
import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from plotmultidim import plot_cluster

'''
root= tk.Tk()

canvas1 = tk.Canvas(root, width = 400, height = 300,  relief = 'raised')
canvas1.pack()

label1 = tk.Label(root, text='k-Means Clustering')
label1.config(font=('helvetica', 14))
canvas1.create_window(200, 25, window=label1)

label2 = tk.Label(root, text='Type Number of Clusters:')
label2.config(font=('helvetica', 8))
canvas1.create_window(200, 120, window=label2)

entry1 = tk.Entry (root)
canvas1.create_window(200, 140, window=entry1)


def getKMeans():
    global numberOfClusters

    Data = pandas.read_csv(r'credit_card_data.csv')
    df = DataFrame(Data, columns=['BALANCE', 'BALANCE_FREQUENCY'])
    numberOfClusters = int(entry1.get())

    kmeans = KMeans(n_clusters=numberOfClusters).fit(df)
    centroids = kmeans.cluster_centers_

    label3 = tk.Label(root, text=centroids)
    canvas1.create_window(200, 250, window=label3)

    figure1 = plt.Figure(figsize=(4, 3), dpi=100)
    ax1 = figure1.add_subplot(111)
    ax1.scatter(df['BALANCE'], df['BALANCE_FREQUENCY'], c=kmeans.labels_.astype(float), s=50, alpha=0.5)
    ax1.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
    scatter1 = FigureCanvasTkAgg(figure1, root)
    scatter1.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)


processButton = tk.Button(text=' Process k-Means ', command=getKMeans, bg='brown', fg='white', font=('helvetica', 10, 'bold'))
canvas1.create_window(200, 170, window=processButton)

root.mainloop()
'''

if __name__ == '__main__':


    '''
    Data = {
        'x': [25, 34, 22, 27, 33, 33, 31, 22, 35, 34, 67, 54, 57, 43, 50, 57, 59, 52, 65, 47, 49, 48, 35, 33, 44, 45,
              38, 43, 51, 46],
        'y': [79, 51, 53, 78, 59, 74, 73, 57, 69, 75, 51, 32, 40, 47, 53, 36, 35, 58, 59, 50, 25, 20, 14, 12, 20, 5, 29,
              27, 8, 7]
    }

    df = DataFrame(Data, columns=['x', 'y'])
'''
    Data = pandas.read_csv(r'credit_card_data.csv')
    df = DataFrame(Data, columns=['BALANCE'])
    df2 = DataFrame(Data, columns=['BALANCE', 'BALANCE_FREQUENCY','PURCHASES','ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'TENURE'])
    df2['CREDIT_LIMIT'] = df2['CREDIT_LIMIT'].fillna(df2['CREDIT_LIMIT'].median())
    print(df2)
    df2['MINIMUM_PAYMENTS'] = df2['MINIMUM_PAYMENTS'].fillna(df2['MINIMUM_PAYMENTS'].median())
    #df2.dropna(subset=['CREDIT_LIMIT'], inplace=True)
    print(df2.isnull().sum())

    wcss=[]
    for i in range(1,20):
        kmeans = KMeans(n_clusters=i,init='k-means++',random_state=101)
        kmeans.fit(df2)
        wcss.append(kmeans.inertia_)

    plt.plot(range(1,20),wcss)
    plt.title('The Elbow Method')
    plt.xlabel('NUmber of clusters')
    plt.ylabel('WCSS')
    plt.show()

    #kmeans = KMeans(n_clusters=4).fit(df)
    kmeans = KMeans(n_clusters=8, init='k-means++', random_state=101).fit(df2)
    centroids = kmeans.cluster_centers_
    print(centroids)
    print(df2['BALANCE'].index.values)
    #plt.scatter(df2['BALANCE'].index.values,df2['BALANCE'],  c=kmeans.labels_.astype(float), s=50, alpha=0.5)
    #plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
    #plt.show()
    plot_cluster(df2.to_numpy(), kmeans.cluster_centers_, kmeans.labels_, 8)


