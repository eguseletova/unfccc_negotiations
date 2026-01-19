import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
mpl.rcParams['font.family'] = 'Times New Roman'
subcomponents_df = pd.read_csv("components.csv")

# 1. select the relevant columns for PCA & VIF
X = subcomponents_df[["text_similarity", "influence_score", "centrality_score", "opposition_factor"]]

# 2. perform PCA with up to 4 components
pca = PCA(n_components=4)
pca.fit(X)

# 3. print explained variance ratio for each principal component
explained_variance = pca.explained_variance_ratio_
print("Explained Variance Ratio per Component:")
for i, ev in enumerate(explained_variance, start=1):
    print(f"Component {i}: {ev:.4f}")
print(f"Cumulative Explained Variance: {np.cumsum(explained_variance)}")

# 5. plot the cumulative explained variance

plt.figure(figsize=(6,4))
plt.plot(
    range(1, 5),
    np.cumsum(explained_variance),
    marker='o',
    markersize=5,
    color='black',
    linewidth=1.5,
    linestyle='-'
)

plt.xlabel('Number of Principal Components', fontsize=11)
plt.ylabel('Cumulative Explained Variance', fontsize=11)
#plt.title('PCA Explained Variance', fontsize=12)

plt.grid(False)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xticks([1,2,3,4], fontsize=10)
plt.yticks(fontsize=10)
plt.ylim(0.75, 1.01)
plt.xlim(0.9, 4.1)

plt.tight_layout()
plt.show()

vif_data = pd.DataFrame()
vif_data["Subcomponent"] = X.columns
vif_data["VIF"] = [
    variance_inflation_factor(X.values, i)
    for i in range(X.shape[1])
]

print("\nVariance Inflation Factor (VIF) for Each Subcomponent:")
print(vif_data)
