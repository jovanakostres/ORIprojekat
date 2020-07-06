import pandas
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tkinter as tk
import tkinter.ttk as ttk
from ttkthemes import ThemedStyle
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from interpretation import cluster_report
from plotmultidim import plot_cluster
from plotmultidim2 import plot_clusters_2d, getElbow, getLabelsKMeans

Data = pandas.read_csv(r'credit_card_data.csv')
print(Data.describe())
df2 = DataFrame(Data, columns=['BALANCE','PURCHASES','ONEOFF_PURCHASES','CASH_ADVANCE', 'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY','CREDIT_LIMIT'])
df2['CREDIT_LIMIT'] = df2['CREDIT_LIMIT'].fillna(df2['CREDIT_LIMIT'].median())

X_df = df2.__deepcopy__()
X_df.sample(5, random_state=0)
pipeline = Pipeline(steps=[
     ('scaler', StandardScaler()),
     ('dim_reduction', PCA(n_components=2, random_state=0))
])

pc = pipeline.fit_transform(X_df)
# pccent = pipeline.fit_transform(kmeans_model.centroid_list)
y_labels = getLabelsKMeans(pc)

root= tk.Tk()
root.title("Credit Cards")
style = ThemedStyle(root)

style.set_theme('arc')
#print(style.theme_names())

canvas1 = tk.Canvas(root, width = 400, height = 300,  relief = 'raised')
canvas1.pack()

label1 = ttk.Label(root, text='k-Means Clustering')
label1.config(font=('helvetica', 14))
canvas1.create_window(200, 25, window=label1)


def getHistogrami():
    # Toplevel object which will
    # be treated as a new window
    newWindow = tk.Toplevel(root)

    # sets the title of the
    # Toplevel widget
    newWindow.title("Analiza")

    # sets the geometry of toplevel
    newWindow.geometry("400x400")

    # A Label widget to show in toplevel
    ttk.Label(newWindow, text="Analiza podataka").pack()
    make_label(newWindow, 10, 20, 380, 350, "Iznos koji je korisnik uplatio unapred (CASH_ADVANCE) i stanje na kartici (BALANCE) nalaze se u korelaciji tj., korisnik koji ima veće stanje će imati tendenciju da plaća iznosom koji je uplatio unapred. Takođe, korisnici koji imaju veće"
                                            "stanje na računu će verovatno imati i veći kreditni limit (CREDITLIMIT)." + "\n" + "Iz histograma vezanog za stanje na kartici možemo videti da korisnici koji aktivno koriste svoje kreditne kartice teže da maksimalno koriste svoje kreditne kartice sve dok stanje ne padne na 0. "
                                            " Histogram ukupnih iznosa potrošnih na kupovinu (PURCHASES) nam kaže da većina korisnika ne koristi svoje kreditne kartice za kupovinu, tek njih oko 40% koristi svoju kreditnu karticu u te svrhe. Limit većine korisnika kreditnih kartica je oko 1000.")

    f, axes = plt.subplots(2, 4, figsize=(30, 30))
    for ax, feature in zip(axes.flat, df2.columns):
        sns.distplot(df2[feature], color="green", ax=ax)
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.show()


histButton = ttk.Button(root, text=' Show Histograms', command=getHistogrami)
canvas1.create_window(200, 120, window=histButton)

processButton = ttk.Button(root, text=' Show Elbow Method ', command=  lambda: getElbow(pc))
canvas1.create_window(200, 150, window=processButton)

clustersButton = ttk.Button(root, text=' Show Clusters in 2D', command= lambda: plot_clusters_2d(df2.__deepcopy__(), y_labels,pc))
canvas1.create_window(200, 180, window=clustersButton)


def openNewWindow():
    # Toplevel object which will
    # be treated as a new window
    newWindow = tk.Toplevel(root)

    # sets the title of the
    # Toplevel widget
    newWindow.title("Analiza")

    # sets the geometry of toplevel
    newWindow.geometry("1200x600")

    # A Label widget to show in toplevel
    ttk.Label(newWindow,
             text="Analiza klastera").pack()

    tab_parent = ttk.Notebook(newWindow)
    tab1 = ttk.Frame(tab_parent)
    tab2 = ttk.Frame(tab_parent)
    tab3 = ttk.Frame(tab_parent)
    tab4 = ttk.Frame(tab_parent)
    tab5 = ttk.Frame(tab_parent)
    tab6 = ttk.Frame(tab_parent)
    tab7 = ttk.Frame(tab_parent)
    tab_parent.add(tab1, text="Prvi klaster")
    tab_parent.add(tab2, text="Drugi klaster")
    tab_parent.add(tab3, text="Treci klaster")
    tab_parent.add(tab4, text="Cetvrti klaster")
    tab_parent.add(tab5, text="Peti klaster")
    tab_parent.add(tab6, text="Sesti klaster")
    tab_parent.add(tab7, text="Sedmi klaster")
    tab_parent.pack(expand=1, fill='both')

    cluster_map = pandas.DataFrame()
    cluster_map['data_index'] = df2.index.values
    cluster_map['cluster'] = y_labels

    make_tab(tab1,cluster_map,0)
    make_tab(tab2, cluster_map, 1)
    make_tab(tab3, cluster_map, 2)
    make_tab(tab4, cluster_map, 3)
    make_tab(tab5, cluster_map, 4)
    make_tab(tab6, cluster_map, 5)
    make_tab(tab7, cluster_map, 6)

    make_label(tab1, 10, 10, 600, 330, text='Korisnici u ovom klasteru retko kupuju. Stanja na računu su niska i odlikuje ih niska potrošnja. Retko kupuju jednokratno. Kada kupuju, kupuju stvari za male pare.')
    make_label(tab2, 10, 10, 600, 330, text='Korisnici u ovom klasteru osrednje često do retko kupuju. Korisnici ove grupe koji imaju nisku količinu novca na racunu retko jednokratno kupuju, ali uplaćuju vece sume novca unapred. Dok korisnici koji imaju osrednju kolicinu novca na računu imaju veci kreditni limit, ali uplaćuju manje sume novca unapred.')
    make_label(tab3, 10, 10, 600, 330, text='Korisnici u ovom klasteru osrednje često do retko kupuju. Stanja novca na računu su visoka i odlikuje ih niska potrošnja tokom kupovine. Kreditni limit i iznosi koje uplaćuju unapred na njihovim karticama su visoki. Predstavljaju osrednje korisnike kreditnih kartica.')
    make_label(tab4, 10, 10, 600, 330, text='Korisnici u ovom klasteru često kupuju, ali retko kupuju jednokratno. Stanja na računu su niska i odlikuje ih niska potrošnja. Kreditni limit je osrednji. Niske uplate iznosa unapred. Kada kupuju, kupuju jeftinije stvari.')
    make_label(tab5, 10, 10, 600, 330, text='Korisnici u ovom klasteru često kupuju u šta spadaju i jednokratne kupovine. Osrednje do nisko stanje novca na kartici i troše velike iznose. Visoki kreditni limit i niski iznosi uplata unapred. Veliki potrošači.')
    make_label(tab6, 10, 10, 600, 330, text='Korisnici u ovom klasteru često kupuju i srednje do veoma često jednokratno kupuju. Stanja na računu su niska i odlikuje ih niski do srednji iznosi potrošnja. Osrednji iznos kreditnog limita i niski iznosi uplata unapred.')
    make_label(tab7, 10, 10, 600, 330, text='Korisnici u ovom klasteru često kupuju u šta spadaju i jednokratne kupovine. Stanja na računu su osrednja i odlikuje ih visoki iznosi potrošnja. Visoki iznos kreditnog limita i niski iznosi uplata unapred.')

clusterReportButton = ttk.Button(root, text='Print Clusters Report', command=lambda: cluster_report(df2, y_labels, 10, 0.01))
canvas1.create_window(200, 210, window=clusterReportButton)


def make_tab(tab,cluster_map,cluster_num):
    figure1 = plt.Figure(figsize=(8.5, 4), dpi=100)
    ax1 = figure1.add_subplot(241)
    ax1.hist(df2[df2.index.isin(cluster_map[cluster_map.cluster == cluster_num]['data_index'])]['BALANCE'])
    ax1.set_title('BALANCE')
    ax2 = figure1.add_subplot(242)
    ax2.hist(df2[df2.index.isin(cluster_map[cluster_map.cluster == cluster_num]['data_index'])]['PURCHASES'])
    ax2.set_title('PURCHASES')
    ax3 = figure1.add_subplot(243)
    ax3.hist(df2[df2.index.isin(cluster_map[cluster_map.cluster == cluster_num]['data_index'])]['ONEOFF_PURCHASES'])
    ax3.set_title('ONEOFF PURCHASES')
    ax4 = figure1.add_subplot(244)
    ax4.hist(df2[df2.index.isin(cluster_map[cluster_map.cluster == cluster_num]['data_index'])]['CASH_ADVANCE'])
    ax4.set_title('CASH ADVANCE')
    ax5 = figure1.add_subplot(245)
    ax5.hist(df2[df2.index.isin(cluster_map[cluster_map.cluster == cluster_num]['data_index'])]['PURCHASES_FREQUENCY'])
    ax5.set_title('P. FREQUENCY')
    ax6 = figure1.add_subplot(246)
    ax6.hist(df2[df2.index.isin(cluster_map[cluster_map.cluster == cluster_num]['data_index'])]['ONEOFF_PURCHASES_FREQUENCY'])
    ax6.set_title('OP FREQUENCY')
    ax7 = figure1.add_subplot(247)
    ax7.hist(df2[df2.index.isin(cluster_map[cluster_map.cluster == cluster_num]['data_index'])]['CREDIT_LIMIT'])
    ax7.set_title('CREDIT LIMIT')
    figure1.set_tight_layout(True)
    scatter1 = FigureCanvasTkAgg(figure1, tab)
    scatter1.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)


def make_label(master, x, y, h, w, text):
    f = ttk.Frame(master, height=h - 60, width=w)
    f.pack_propagate(0) # don't shrink
    f.place(x=x, y=y)
    label = tk.Message(f, text=text, width=w - 10, anchor='n' )
    label.configure(font=("Calibri", 12, "normal"))
    label.pack(fill=tk.BOTH, expand=1)
    return label

def printNesto():
    print(y_labels)


prntButton = ttk.Button(root, text=' Cluster Analysis', command=openNewWindow)
canvas1.create_window(200, 240, window=prntButton)

root.mainloop()


'''
if __name__ == '__main__':



    Data = {
        'x': [25, 34, 22, 27, 33, 33, 31, 22, 35, 34, 67, 54, 57, 43, 50, 57, 59, 52, 65, 47, 49, 48, 35, 33, 44, 45,
              38, 43, 51, 46],
        'y': [79, 51, 53, 78, 59, 74, 73, 57, 69, 75, 51, 32, 40, 47, 53, 36, 35, 58, 59, 50, 25, 20, 14, 12, 20, 5, 29,
              27, 8, 7]
    }

    df = DataFrame(Data, columns=['x', 'y'])

    Data = pandas.read_csv(r'credit_card_data.csv')
    print(Data.describe())
    df = DataFrame(Data, columns=['BALANCE'])
    df2 = DataFrame(Data, columns=['BALANCE','PURCHASES','ONEOFF_PURCHASES','CASH_ADVANCE', 'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY','CREDIT_LIMIT'])
    df2['CREDIT_LIMIT'] = df2['CREDIT_LIMIT'].fillna(df2['CREDIT_LIMIT'].median())
    #print(df2)
    #df2['MINIMUM_PAYMENTS'] = df2['MINIMUM_PAYMENTS'].fillna(df2['MINIMUM_PAYMENTS'].median())
    #df2.dropna(subset=['CREDIT_LIMIT'], inplace=True)
    print(df2.isnull().sum())


    

    #getElbow()

    #kmeans = KMeans(n_clusters=4).fit(df)
    #kmeans = KMeans(n_clusters=8, init='k-means++', random_state=101).fit(df2)
    #centroids = kmeans.cluster_centers_
    #print(centroids)
    #print(df2['BALANCE'].index.values)
    #plt.scatter(df2['BALANCE'].index.values,df2['BALANCE'],  c=kmeans.labels_.astype(float), s=50, alpha=0.5)
    #plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
    #plt.show()
    #plot_cluster(df2.to_numpy(), kmeans.cluster_centers_, kmeans.labels_, 8)

    y_cl = plot_clusters_2d(df2.__deepcopy__())
    cluster_report(df2, y_cl.labels_,10,0.01)
    
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
'''

