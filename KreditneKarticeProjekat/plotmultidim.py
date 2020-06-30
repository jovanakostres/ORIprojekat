import numpy as np
import colorsys
import random
import os
from matplotlib import pyplot as plt


def get_colors(num_colors):
    """
    Function to generate a list of randomly generated colors
    The function first generates 256 different colors and then
    we randomly select the number of colors required from it
    num_colors        -> Number of colors to generate
    colors            -> Consists of 256 different colors
    random_colors     -> Randomly returns required(num_color) colors
    """
    colors = []
    random_colors = []
    # Generate 256 different colors and choose num_clors randomly
    for i in np.arange(0., 360., 360. / 256.):
        hue = i / 360.
        lightness = (50 + np.random.rand() * 10) / 100.
        saturation = (90 + np.random.rand() * 10) / 100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))

    for i in range(0, num_colors):
        random_colors.append(colors[random.randint(0, len(colors) - 1)])
    return random_colors


def random_centroid_selector(total_clusters , clusters_plotted):
    """
    Function to generate a list of randomly selected
    centroids to plot on the output png
    total_clusters        -> Total number of clusters
    clusters_plotted      -> Number of clusters to plot
    random_list           -> Contains the index of clusters
                             to be plotted
    """
    random_list = []
    for i in range(0 , clusters_plotted):
        random_list.append(random.randint(0, total_clusters - 1))
    return random_list


def plot_cluster(kmeansdata, centroid_list, label_list , num_cluster):
    """
    Function to convert the n-dimensional cluster to
    2-dimensional cluster and plotting 50 random clusters
    file%d.png    -> file where the output is stored indexed
                     by first available file index
                     e.g. file1.png , file2.png ...
    """
    mlab_pca = PCA(kmeansdata)
    cutoff = mlab_pca.fracs[1]
    users_2d = mlab_pca.project(kmeansdata, minfrac=cutoff)
    centroids_2d = mlab_pca.project(centroid_list, minfrac=cutoff)

    colors = get_colors(num_cluster)
    plt.figure()
    plt.xlim([users_2d[:, 0].min() - 3, users_2d[:, 0].max() + 3])
    plt.ylim([users_2d[:, 1].min() - 3, users_2d[:, 1].max() + 3])

    # Plotting 50 clusters only for now
    random_list = random_centroid_selector(num_cluster , 50)

    # Plotting only the points whose centers were plotted
    # Points are represented as a small '+' marker
    for i, position in enumerate(label_list):
        if position in random_list:
            plt.scatter(users_2d[i, 0], users_2d[i, 1] , marker='+' , color=colors[position])

    # Plotting only the centroids which were randomly_selected
    # Centroids are represented as a large 'o' marker
    for i, position in enumerate(centroids_2d):
        if i in random_list:
            plt.scatter(centroids_2d[i, 0], centroids_2d[i, 1], marker='o', color=colors[i], s=100, edgecolors='black')

    plt.show()
    '''
    filename = "name"
    i = 0
    while True:
        if os.path.isfile(filename + str(i) + ".png") == False:
            #new index found write file and return
            plt.savefig(filename + str(i) + ".png")
            break
        else:
            #Changing index to next number
            i = i + 1     
    '''

    return


class PCA(object):
    def __init__(self, a, standardize=True):
        """
        compute the SVD of a and store data for PCA.  Use project to
        project the data onto a reduced set of dimensions

        Parameters
        ----------
        a : np.ndarray
            A numobservations x numdims array
        standardize : bool
            True if input data are to be standardized. If False, only centering
            will be carried out.

        Attributes
        ----------
        a
            A centered unit sigma version of input ``a``.

        numrows, numcols
            The dimensions of ``a``.

        mu
            A numdims array of means of ``a``. This is the vector that points
            to the origin of PCA space.

        sigma
            A numdims array of standard deviation of ``a``.

        fracs
            The proportion of variance of each of the principal components.

        s
            The actual eigenvalues of the decomposition.

        Wt
            The weight vector for projecting a numdims point or array into
            PCA space.

        Y
            A projected into PCA space.

        Notes
        -----
        The factor loadings are in the ``Wt`` factor, i.e., the factor loadings
        for the first principal component are given by ``Wt[0]``. This row is
        also the first eigenvector.

        """
        n, m = a.shape
        if n < m:
            raise RuntimeError('we assume data in a is organized with '
                               'numrows>numcols')

        self.numrows, self.numcols = n, m
        self.mu = a.mean(axis=0)
        self.sigma = a.std(axis=0)
        self.standardize = standardize

        a = self.center(a)

        self.a = a

        U, s, Vh = np.linalg.svd(a, full_matrices=False)

        # Note: .H indicates the conjugate transposed / Hermitian.

        # The SVD is commonly written as a = U s V.H.
        # If U is a unitary matrix, it means that it satisfies U.H = inv(U).

        # The rows of Vh are the eigenvectors of a.H a.
        # The columns of U are the eigenvectors of a a.H.
        # For row i in Vh and column i in U, the corresponding eigenvalue is
        # s[i]**2.

        self.Wt = Vh

        # save the transposed coordinates
        Y = np.dot(Vh, a.T).T
        self.Y = Y

        # save the eigenvalues
        self.s = s**2

        # and now the contribution of the individual components
        vars = self.s / len(s)
        self.fracs = vars/vars.sum()

    def project(self, x, minfrac=0.):
        '''
        project x onto the principle axes, dropping any axes where fraction
        of variance<minfrac
        '''
        x = np.asarray(x)
        if x.shape[-1] != self.numcols:
            raise ValueError('Expected an array with dims[-1]==%d' %
                             self.numcols)
        Y = np.dot(self.Wt, self.center(x).T).T
        mask = self.fracs >= minfrac
        if x.ndim == 2:
            Yreduced = Y[:, mask]
        else:
            Yreduced = Y[mask]
        return Yreduced


    def center(self, x):
        '''
        center and optionally standardize the data using the mean and sigma
        from training set a
        '''
        if self.standardize:
            return (x - self.mu)/self.sigma
        else:
            return (x - self.mu)


    @staticmethod
    def _get_colinear():
        c0 = np.array([
            0.19294738,  0.6202667,   0.45962655,  0.07608613,  0.135818,
            0.83580842,  0.07218851,  0.48318321,  0.84472463,  0.18348462,
            0.81585306,  0.96923926,  0.12835919,  0.35075355,  0.15807861,
            0.837437,    0.10824303,  0.1723387,   0.43926494,  0.83705486])

        c1 = np.array([
            -1.17705601, -0.513883,   -0.26614584,  0.88067144,  1.00474954,
            -1.1616545,   0.0266109,   0.38227157,  1.80489433,  0.21472396,
            -1.41920399, -2.08158544, -0.10559009,  1.68999268,  0.34847107,
            -0.4685737,   1.23980423, -0.14638744, -0.35907697,  0.22442616])

        c2 = c0 + 2*c1
        c3 = -3*c0 + 4*c1
        a = np.array([c3, c0, c1, c2]).T
        return a