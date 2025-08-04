# Financial Markets Manifold Learning & Topological Analysis

This repository contains Python code and data for analyzing the topological structure of financial markets using manifold learning and persistent homology. It includes scripts for:

- Visualizing manifold embeddings of S&P 500 stocks
- Evaluating the topological preservation (TopoPres) under Gaussian noise

---

## 📁 Repository Structure

📂 data/
├── sp500_log_returns.csv # Daily log returns of S&P 500 stocks
├── sp500_filtered_close.csv # Filtered closing prices with sector labels
├── sp500_companies.csv # Company names and sectors
└── sectors.csv # Sector names and IDs

📜 embedding_visualization.py # Sector-based 2D embedding using t-SNE, UMAP, Isomap
📜 topopres_sensitivity_analysis.py # Sensitivity analysis of TopoPres under noise
📊 financial_embeddings.png # 2D embeddings colored by sector
📊 topopres_sensitivity.png # TopoPres change across noise levels


---

## 🧪 1. Manifold Embedding & Sector Visualization

**Script:** `embedding_visualization.py`

This script performs 2D manifold embeddings of the S&P 500 stock return data using:
- **t-SNE**
- **UMAP**
- **Isomap**

Each stock is colored by its sector label to visually assess cluster separation and manifold structure.

> **Input Files:**
- `data/sp500_log_returns.csv`: Log returns (rows = dates, columns = stocks)
- `data/sp500_filtered_close.csv`: Must contain a `sector` column with industry labels

> **Output:**
- `financial_embeddings.png`: 2D scatter plots colored by sector

---

## 🔬 2. TopoPres Sensitivity Analysis

**Script:** `topopres_sensitivity_analysis.py`

This script investigates the stability of **TopoPres** under added Gaussian noise. The steps include:

1. Compute baseline TopoPres scores from log-return data
2. Add Gaussian noise with increasing standard deviation (σ = 0.01, 0.02, 0.05)
3. Recalculate TopoPres for each noisy version
4. Measure deviation from the baseline for:
   - Isomap
   - t-SNE
   - UMAP
5. Perform statistical analysis (mean ± std, one-way ANOVA)

> **Output:**
- `topopres_sensitivity.png`: Average TopoPres deviation across noise levels
- Summary printed to console (mean ± std)
- ANOVA results shown in terminal

---

## 📊 Sample Results

Example figure:  
`financial_embeddings.png`  
Shows sector-wise separation in 2D embeddings.

Example figure:  
`topopres_sensitivity.png`  
Displays TopoPres deviations under different noise levels.

---

## 🔧 Installation

Install required dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn umap-learn ripser persim scipy
