from data import data_generator
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt

number_of_clusters = 3
number_of_tnse_components = 2

is_synthetic = False

if is_synthetic:
    gmm = data_generator.GaussianMixtureData(dimension=32,
                                             number_of_clusters=number_of_clusters,
                                             tightness=0.1,
                                             cube_side_length=100)
    data = gmm.data
    labels = gmm.labelled_data[:, 0]
else:
    from numpy import genfromtxt
    data = genfromtxt('../data/cleveland_data.csv', delimiter=',')
    labels = data[1:, 1]
    data = data[1:, 2:]

perplexity_list = [5, 10, 30, 50, 100]
for perplexity in perplexity_list:
    embedded_data = TSNE(n_components=number_of_tnse_components, perplexity=perplexity).fit_transform(data)

    colors = ['#00B7BA', '#FFB86F', '#5E6572', '#6B0504', '#BA5C12']
        # Tiffany blue, Very light tangelo, Cadet, Rosewood, Alloy orange
    fig_3d = plt.figure(dpi=200)

    if number_of_tnse_components == 2:
        ax = fig_3d.add_subplot()
        ax.scatter(embedded_data[:, 0],
                   embedded_data[:, 1],
                   c=labels,
                   cmap=matplotlib.colors.ListedColormap(colors))

    if number_of_tnse_components == 3:
        ax = fig_3d.add_subplot(projection='3d')
        ax.scatter(embedded_data[:, 0],
                   embedded_data[:, 1],
                   embedded_data[:, 2],
                   c=labels,
                   cmap=matplotlib.colors.ListedColormap(colors))

    ax.set_title('t-SNE with Perplexity {}'.format(perplexity))
    filename = 'adni_tsne_{}.png'.format(perplexity)
    fig_3d.savefig(filename)

plt.show()
