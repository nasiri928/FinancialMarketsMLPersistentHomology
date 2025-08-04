"""
embedding_visualization.py

This script visualizes the low-dimensional embeddings of financial market data using
three popular manifold learning methods: t-SNE, UMAP, and Isomap. The stocks are colored
based on their sector labels to assess the clustering and separation of sectors.

Required input files:
- sp500_filtered_close.csv: Contains closing prices and optionally sector labels.
- sp500_log_returns.csv: Contains the log returns computed from the closing prices.

All input files must be in the same directory as this script.

Author: Mohammad Nasiri
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE, Isomap
import umap
from sklearn.preprocessing import StandardScaler

# === 1. Load data ===
close_df = pd.read_csv('sp500_filtered_close.csv', index_col=0)
returns_df = pd.read_csv('sp500_log_returns.csv', index_col=0)

# === 2. Prepare data ===
X = returns_df.T  # Transpose: rows = stocks, columns = time points
stock_names = X.index.tolist()

# === 2.5 Sector labels mapping ===
if 'sector' in close_df.columns:
    sector_series = close_df['sector']
    sector_series = sector_series.loc[sector_series.index.intersection(X.index)]
    labels = sector_series.reindex(X.index).fillna('Unknown').values
else:
    labels = np.array(['Unknown'] * len(X))

# === 3. Normalize data ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 4. Define embedding methods ===
embedders = {
    "t-SNE": TSNE(n_components=2, perplexity=30, random_state=42),
    "UMAP": umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42),
    "Isomap": Isomap(n_components=2, n_neighbors=10)
}

# === 5. Color map for sectors ===
unique_labels = np.unique(labels)
palette = sns.color_palette("tab20", len(unique_labels))
color_dict = {lab: palette[i] for i, lab in enumerate(unique_labels)}
colors = [color_dict[lab] for lab in labels]

# === 6. Plotting ===
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, embedder) in zip(axes, embedders.items()):
    embedding = embedder.fit_transform(X_scaled)
    for lab in unique_labels:
        ix = labels == lab
        ax.scatter(embedding[ix, 0], embedding[ix, 1], 
                   label=lab, s=15, alpha=0.8, c=[color_dict[lab]])
    ax.set_title(f"{name} Embedding (Sector Coloring)", fontsize=12)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True)

axes[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Sectors")
plt.suptitle("Financial Market Embeddings via Manifold Learning", fontsize=14)
plt.tight_layout()
plt.savefig("financial_embeddings.png", dpi=300, bbox_inches='tight')
plt.show()
