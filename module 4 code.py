r"""RFM customer segmentation, refactored from the Module 4 notebook.

Install: python -m pip install pandas numpy matplotlib scikit-learn
Run: python customer_segmentation_rewritten.py
Default input: C:\Users\kartik\Desktop\DSML PROJECT 4\OnlineRetail.csv
Outputs are saved beside the script. Use --data to select a different CSV.
For exact reproducibility, use the same package versions as the original run.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

RFM_COLUMNS = ["Recency", "Frequency", "Monetary"]


def load_transactions(csv_path):
    transactions = pd.read_csv(csv_path, encoding="ISO-8859-1")
    print("Dataset shape:", transactions.shape)
    print(transactions.head().to_string(index=False))
    print("\nMissing values:\n", transactions.isna().sum())
    return transactions


def prepare_transactions(transactions):
    transactions = transactions.copy()
    transactions["InvoiceDate"] = pd.to_datetime(
        transactions["InvoiceDate"], errors="coerce"
    )
    valid = (
        transactions["CustomerID"].notna()
        & transactions["InvoiceDate"].notna()
        & ~transactions["InvoiceNo"].astype(str).str.startswith("C")
        & transactions["Quantity"].gt(0)
        & transactions["UnitPrice"].gt(0)
    )
    purchases = transactions.loc[valid].copy()
    purchases["CustomerID"] = purchases["CustomerID"].astype(int)
    purchases["Sales"] = purchases["Quantity"].mul(purchases["UnitPrice"])
    print("\nRows after cleaning:", len(purchases))
    print("Unique customers:", purchases["CustomerID"].nunique())
    print("Latest transaction date:", purchases["InvoiceDate"].max())
    return purchases


def calculate_rfm(purchases):
    reference_day = purchases["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
    grouped = purchases.groupby("CustomerID", sort=True)
    customer_metrics = pd.DataFrame({
        "Recency": (reference_day - grouped["InvoiceDate"].max()).dt.days,
        "Frequency": grouped["InvoiceNo"].nunique(),
        "Monetary": grouped["Sales"].sum(),
    }).reset_index()
    print("\nSnapshot date:", reference_day)
    print("Customers in RFM table:", len(customer_metrics))
    print(customer_metrics.head().to_string(index=False))
    print(customer_metrics[RFM_COLUMNS].describe().round(2))
    return customer_metrics


def assign_clusters(customer_metrics):
    transformed = np.log1p(customer_metrics[RFM_COLUMNS])
    scaled = StandardScaler().fit_transform(transformed)
    model = KMeans(n_clusters=4, random_state=42, n_init=10)
    segmented = customer_metrics.copy()
    segmented["Cluster"] = model.fit_predict(scaled)
    score = silhouette_score(scaled, segmented["Cluster"])
    print(f"\nSilhouette score: {score:.3f}")
    return segmented


def summarise_clusters(segmented):
    summary = segmented.groupby("Cluster").agg(
        Customers=("CustomerID", "count"),
        Median_Recency=("Recency", "median"),
        Median_Frequency=("Frequency", "median"),
        Median_Monetary=("Monetary", "median"),
        Total_Revenue=("Monetary", "sum"),
    ).reset_index()
    summary["Customer_Share_%"] = summary["Customers"].div(len(segmented)).mul(100).round(1)
    summary["Revenue_Share_%"] = (
        summary["Total_Revenue"].div(segmented["Monetary"].sum()).mul(100).round(1)
    )
    print("\nSegment summary:\n", summary.round(2).to_string(index=False))
    return summary


def plot_clusters(summary, output_dir):
    figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    chart_settings = [
        ("Customers", "Customers in each cluster", "Number of customers"),
        ("Total_Revenue", "Total revenue by cluster", "Revenue"),
    ]
    for axis, (column, title, ylabel) in zip(axes, chart_settings):
        summary.plot.bar(x="Cluster", y=column, ax=axis, legend=False, title=title)
        axis.set(xlabel="Cluster", ylabel=ylabel)
    figure.tight_layout()
    figure.savefig(output_dir / "customer_cluster_charts.png", dpi=150)
    plt.show()
    plt.close(figure)


def main():
    output_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="RFM customer segmentation")
    parser.add_argument(
        "--data", type=Path,
        default=Path(r"C:\Users\kartik\Desktop\DSML PROJECT 4\OnlineRetail.csv"),
        help="Path to the Online Retail CSV",
    )
    args = parser.parse_args()
    transactions = load_transactions(args.data)
    purchases = prepare_transactions(transactions)
    customer_metrics = calculate_rfm(purchases)
    segmented = assign_clusters(customer_metrics)
    summary = summarise_clusters(segmented)
    segmented.to_csv(output_dir / "customer_segments.csv", index=False)
    summary.to_csv(output_dir / "segment_summary.csv", index=False)
    print("\nSaved: customer_segments.csv and segment_summary.csv")
    # Cluster IDs are arbitrary. Interpret them from the median RFM values.
    plot_clusters(summary, output_dir)


if __name__ == "__main__":
    main()
