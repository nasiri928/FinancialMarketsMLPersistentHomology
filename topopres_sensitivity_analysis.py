"""
Title: Sensitivity Analysis of TopoPres under Gaussian Noise for Isomap, t-SNE, and UMAP

Description:
This script analyzes the robustness of the Topological Preservation (TopoPres) metric under Gaussian noise perturbations
using three manifold learning techniques: Isomap, t-SNE, and UMAP. 

Main Steps:
1. Load SP500 log-return data.
2. Compute baseline TopoPres for each method.
3. Add Gaussian noise at multiple levels (σ = 0.01, 0.02, 0.05).
4. Recompute TopoPres and measure deviation.
5. Plot sensitivity curves.
6. Compute descriptive statistics (mean ± std).
7. Conduct ANOVA to test significance of differences across methods.

Requirements:
- numpy, pandas, matplotlib, sklearn, umap-learn, ripser, persim, scipy

Data:
This script expects the file: ./data/sp500_log_returns.csv

Author: Mohammad Nasiri
GitHub Repository: https://github.com/nasiri928/FinancialMarketsMLPersistentHomology
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap, TSNE
import umap
from scipy.spatial.distance import pdist, squareform
from ripser import ripser
from persim import bottleneck
from scipy.stats import f_oneway
import warnings
import pprint

warnings.filterwarnings('ignore')

# ---------------- Configuration ----------------
window_size = 90
stride = 30
noise_levels = [0.01, 0.02, 0.05]
methods = ['isomap', 'tsne', 'umap']

# ---------------- Load Data ----------------
log_returns = pd.read_csv("./data/sp500_log_returns.csv", index_col=0)

# ---------------- Embedding Computation ----------------
def compute_embedding(data, method):
    if method == 'isomap':
        dist = squareform(pdist(data, metric='euclidean'))
        model = Isomap(n_neighbors=10, n_components=2, metric='precomputed')
        embedding = model.fit_transform(dist)
    elif method == 'umap':
        dist = squareform(pdist(data, metric='euclidean'))
        model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='precomputed', random_state=42)
        embedding = model.fit_transform(dist)
    elif method == 'tsne':
        model = TSNE(n_components=2, perplexity=50, n_iter=1000, learning_rate=200, random_state=42)
        embedding = model.fit_transform(data.values)
    return embedding

# ---------------- Persistent Homology ----------------
def compute_pd(X):
    result = ripser(X, maxdim=1)
    return result['dgms']

# ---------------- TopoPres Calculation ----------------
def compute_topopres(pd_orig, pd_embed):
    d0 = bottleneck(pd_orig[0], pd_embed[0])
    d1 = bottleneck(pd_orig[1], pd_embed[1])
    return 1 - 0.5 * (d0 + d1)

# ---------------- Sensitivity Analysis ----------------
def topopres_sensitivity(log_returns):
    base_scores = {m: [] for m in methods}
    deviations = {m: [] for m in methods}
    
    for start in range(0, len(log_returns) - window_size + 1, stride):
        window = log_returns.iloc[start:start+window_size]
        pd_orig = compute_pd(window.values)
        
        for method in methods:
            emb = compute_embedding(window, method)
            pd_emb = compute_pd(emb)
            score = compute_topopres(pd_orig, pd_emb)
            base_scores[method].append(score)

    for sigma in noise_levels:
        for start in range(0, len(log_returns) - window_size + 1, stride):
            window = log_returns.iloc[start:start+window_size]
            noisy_window = window + np.random.normal(0, sigma, window.shape)
            pd_orig = compute_pd(window.values)

            for method in methods:
                emb = compute_embedding(noisy_window, method)
                pd_emb = compute_pd(emb)
                score = compute_topopres(pd_orig, pd_emb)
                delta = abs(score - base_scores[method][start // stride])
                deviations[method].append((sigma, delta))

    return deviations

# ---------------- Run Analysis ----------------
devs = topopres_sensitivity(log_returns)

# ---------------- Plotting ----------------
plt.figure(figsize=(8,6))
for method in methods:
    sigmas = [x[0] for x in devs[method]]
    deltas = [x[1] for x in devs[method]]
    means = [np.mean([deltas[i] for i in range(len(deltas)) if sigmas[i]==s]) for s in noise_levels]
    plt.plot(noise_levels, means, marker='o', label=method.upper())

plt.xlabel("Noise Level (σ)")
plt.ylabel("Average |ΔTopoPres|")
plt.title("Sensitivity of TopoPres to Gaussian Noise")
plt.legend()
plt.grid(True) 
plt.tight_layout()
plt.savefig("topopres_sensitivity.png", dpi=300)
plt.show()

# ---------------- Descriptive Statistics ----------------
summary_stats = {}

for method in methods:
    sigmas = [x[0] for x in devs[method]]
    deltas = [x[1] for x in devs[method]]
    summary_stats[method] = {}

    for sigma in noise_levels:
        sigma_deltas = [deltas[i] for i in range(len(deltas)) if sigmas[i] == sigma]
        mean = np.mean(sigma_deltas)
        std = np.std(sigma_deltas)
        summary_stats[method][sigma] = (round(mean, 4), round(std, 4))

print("\nSummary Statistics (mean ± std) for each method and noise level:")
pprint.pprint(summary_stats)

# ---------------- ANOVA Test ----------------
for sigma in noise_levels:
    print(f"\nANOVA for σ = {sigma}:")
    samples = []
    for method in methods:
        sigmas = [x[0] for x in devs[method]]
        deltas = [x[1] for x in devs[method]]
        sigma_deltas = [deltas[i] for i in range(len(deltas)) if sigmas[i] == sigma]
        samples.append(sigma_deltas)
    fval, pval = f_oneway(*samples)
    print(f"F-statistic = {fval:.3f}, p-value = {pval:.4f}")
