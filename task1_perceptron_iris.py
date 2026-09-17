import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report

class Perceptron:
    def __init__(self, add_bias=True, max_iters=10000, record_updates=False):
        self.max_iters = max_iters
        self.add_bias = add_bias
        self.record_updates = record_updates
        if record_updates:
            self.w_hist = []
            self.n_hist = []

    def fit(self, x, y):
        if x.ndim == 1:
            x = x[:, None]
        if self.add_bias:
            n_samples = x.shape[0]
            x = np.column_stack([x, np.ones(n_samples)])

        n_samples, n_features = x.shape
        w = np.zeros(n_features)

        if self.record_updates:
            self.w_hist = [w.copy()]

        y_signed = 2 * y - 1

        t = 0
        change = True

        while change and t < self.max_iters:
            change = False
            for n in np.random.permutation(n_samples):
                yh = np.sign(np.dot(x[n, :], w))
                if yh == y_signed[n]:
                    continue
                w = w + y_signed[n] * x[n, :]
                if self.record_updates:
                    self.w_hist.append(w.copy())
                    self.n_hist.append(n)
                change = True
                t += 1
                if t >= self.max_iters:
                    break

        if change:
            print(f"did not converge after {t} updates")
        else:
            print(f"converged after {t} iterations!")

        self.w = w
        return self

    def predict(self, x):
        if x.ndim == 1:
            x = x[:, None]
        n_test = x.shape[0]
        if self.add_bias:
            x = np.column_stack([x, np.ones(n_test)])
        yh = np.sign(np.dot(x, self.w))
        return ((yh + 1) // 2).astype(int)


def main():
    iris = load_iris()
    X_all = iris["data"]
    y_all = iris["target"]

    mask = (y_all == 0) | (y_all == 2)
    X = X_all[mask]
    y_raw = y_all[mask]

    X = X[:, [0, 2]]

    y = (y_raw == 2).astype(int)

    model = Perceptron(record_updates=True)
    model.fit(X, y)

    y_pred = model.predict(X)

    print("\nClassification report for Perceptron (setosa vs virginica):")
    print(classification_report(y, y_pred, target_names=["setosa", "virginica"]))

    plt.figure(figsize=(8, 6))

    plt.scatter(
        X[y == 0, 0],
        X[y == 0, 1],
        c="blue",
        marker="o",
        label="setosa"
    )
    plt.scatter(
        X[y == 1, 0],
        X[y == 1, 1],
        c="red",
        marker="s",
        label="virginica"
    )

    w = model.w
    x_line = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 200)

    if w[1] != 0:
        coef = -w[0] / w[1]
        intercept = -w[2] / w[1]
        y_line = coef * x_line + intercept
        plt.plot(x_line, y_line, "k-", label="decision boundary")

    plt.xlabel("sepal length (cm)")
    plt.ylabel("petal length (cm)")
    plt.title("Perceptron: Iris setosa vs Iris virginica\nFeatures: sepal length, petal length")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
