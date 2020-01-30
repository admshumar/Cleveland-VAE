from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from numpy import genfromtxt
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


adni_data = genfromtxt('../data/cleveland_data.csv', delimiter=',')


def pca_plot(original_data, number_of_components=10, impairment_class=0, first_principal_component=0):
    if impairment_class > 0:
        data = original_data[original_data[:, 1] == impairment_class]
        data = data[:, 2:]
    else:
        data = original_data[1:, 2:]
    data = StandardScaler().fit_transform(data[:, 2:])
    pca = PCA(n_components=number_of_components)
    pca.fit(data)
    data = pca.transform(data)

    print("PRINCIPAL COMPONENT ANALYSIS")
    print("Variance Ratios:\n", pca.explained_variance_ratio_)
    print("Variance Ratio Sum:\n", sum(pca.explained_variance_ratio_))

    fig = plt.figure(1, figsize=(8, 6))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    colors = ['#0C1B33', '#7A306C', '#03B5AA']

    if impairment_class > 0:
        ax.scatter(data[:, first_principal_component+0],
                   data[:, first_principal_component+1],
                   data[:, first_principal_component+2])
        ax.set_title(f'First three principal components\nImpairment Class: {impairment_class}')
    else:
        ax.scatter(data[:, first_principal_component+0],
                   data[:, first_principal_component+1],
                   data[:, first_principal_component+2],
                   c=original_data[1:, 1],
                   cmap=matplotlib.colors.ListedColormap(colors))
        ax.set_title("PCA of ADNI Data")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_zlabel("Principal Component 3")
    plt.show()


pca_plot(adni_data, impairment_class=0)


#print("Principal Components:", pca.components_)
#print("Singular Values:", pca.singular_values_)
