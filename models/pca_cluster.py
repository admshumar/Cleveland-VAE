from sklearn.decomposition import PCA
from numpy import genfromtxt

data = genfromtxt('../data/cleveland_data.csv', delimiter=',')

pca = PCA(n_components=10)
pca.fit(data[1:, 2:])
print("PRINCIPAL COMPONENT ANALYSIS")
print("Variance Ratios:", pca.explained_variance_ratio_)
print("Variance Ratio Sum:", sum(pca.explained_variance_ratio_))
print("Principal Components:", pca.components_)
print("Singular Values:", pca.singular_values_)
