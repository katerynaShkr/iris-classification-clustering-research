import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from sklearn import preprocessing
from scipy.cluster.hierarchy import linkage, dendrogram

def main():
    iris = datasets.load_iris()
    X = iris.data
    y_true = iris.target

    linkages = ["ward", "complete", "average"]
    print("Adjusted Rand Index without normalization:")
    scores_raw = {}
    for link in linkages:
        model = AgglomerativeClustering(n_clusters=3, linkage=link)
        y_pred = model.fit_predict(X)
        score = adjusted_rand_score(y_true, y_pred)
        scores_raw[link] = score
        print(f"{link}: {score:.4f}")

    normalized_X = preprocessing.normalize(X)

    print("\nAdjusted Rand Index with normalization:")
    scores_norm = {}
    for link in linkages:
        model = AgglomerativeClustering(n_clusters=3, linkage=link)
        y_pred = model.fit_predict(normalized_X)
        score = adjusted_rand_score(y_true, y_pred)
        scores_norm[link] = score
        print(f"{link}: {score:.4f}")

    best_linkage = max(scores_norm, key=scores_norm.get)
    print(f"\nBest linkage on normalized data: {best_linkage} "
          f"(ARI = {scores_norm[best_linkage]:.4f})")

    linkage_matrix = linkage(normalized_X, method=best_linkage)

    plt.figure(figsize=(14, 8))
    dendrogram(linkage_matrix, labels=y_true)
    plt.title(f"Dendrogram for Iris dataset (linkage = {best_linkage})")
    plt.xlabel("Sample index (true class as label)")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
