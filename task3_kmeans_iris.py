import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

def main():
    iris = load_iris()
    X_all = iris["data"]
    y_all = iris["target"]
    target_names = iris["target_names"]

    mask = (y_all == 0) | (y_all == 1)
    X = X_all[mask]
    y_true = y_all[mask]

    X = X[:, [2, 3]]

    wcss = []
    k_values = range(1, 7)
    for k in k_values:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(8, 6))
    plt.plot(k_values, wcss, marker="o")
    plt.xticks(k_values)
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Within-Cluster Sum of Squares (WCSS)")
    plt.title("Elbow method for K-means (Iris: setosa & versicolor)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    k_opt = 2
    kmeans_final = KMeans(n_clusters=k_opt, n_init=10, random_state=42)
    cluster_labels = kmeans_final.fit_predict(X)

    ari = adjusted_rand_score(y_true, cluster_labels)
    print(f"Adjusted Rand Index for K-means (k=2): {ari:.4f}")

    df = pd.DataFrame(X, columns=["petal length", "petal width"])
    df["true_class"] = y_true
    df["cluster"] = cluster_labels

    plt.figure(figsize=(8, 6))
    for cluster_id in range(k_opt):
        cluster_points = df[df["cluster"] == cluster_id]
        plt.scatter(
            cluster_points["petal length"],
            cluster_points["petal width"],
            label=f"Cluster {cluster_id}"
        )

    plt.xlabel("petal length (cm)")
    plt.ylabel("petal width (cm)")
    plt.title("K-means clustering (Iris setosa & versicolor)\nFeatures: petal length, petal width")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
