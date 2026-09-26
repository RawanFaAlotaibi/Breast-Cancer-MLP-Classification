import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def load_data(filepath="wdbc.data"):
    data = []
    labels = []

    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split(",")

            label = 1 if parts[1] == "M" else 0
            features = [float(x) for x in parts[2:]]

            labels.append(label)
            data.append(features)

    return np.array(data, dtype=float), np.array(labels, dtype=int)


def standardize_manual(X_train, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0

    return (X_train - mean) / std, (X_test - mean) / std


def manual_accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


def manual_precision(y_true, y_pred):
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))

    return TP / (TP + FP) if (TP + FP) > 0 else 0.0


def manual_recall(y_true, y_pred):
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FN = np.sum((y_pred == 0) & (y_true == 1))

    return TP / (TP + FN) if (TP + FN) > 0 else 0.0


def manual_f1(y_true, y_pred):
    p = manual_precision(y_true, y_pred)
    r = manual_recall(y_true, y_pred)

    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


class MLP:
    def __init__(self, input_size, hidden_size, learning_rate):
        self.lr = learning_rate

        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, 1) * 0.01
        self.b2 = np.zeros((1, 1))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.y_hat = self.sigmoid(self.z2)

        return self.y_hat

    def backward(self, X, y):
        m = X.shape[0]
        y = y.reshape(-1, 1)

        dz2 = 2 * (self.y_hat - y) * self.y_hat * (1 - self.y_hat)
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        dz1 = np.dot(dz2, self.W2.T) * self.a1 * (1 - self.a1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def predict(self, X):
        return (self.forward(X) >= 0.5).astype(int).ravel()


def train_model(model, X_train, y_train, X_test, y_test, epochs, batch_size=32):
    train_loss_hist = []
    test_loss_hist = []
    train_acc_hist = []
    test_acc_hist = []

    n = X_train.shape[0]

    for epoch in range(epochs):
        indices = np.random.permutation(n)

        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        for start in range(0, n, batch_size):
            end = start + batch_size

            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            model.forward(X_batch)
            model.backward(X_batch, y_batch)

        y_hat_train = model.forward(X_train)
        y_hat_test = model.forward(X_test)

        train_loss_hist.append(
            np.mean((y_hat_train - y_train.reshape(-1, 1)) ** 2)
        )

        test_loss_hist.append(
            np.mean((y_hat_test - y_test.reshape(-1, 1)) ** 2)
        )

        train_acc_hist.append(
            manual_accuracy(
                y_train,
                (y_hat_train >= 0.5).astype(int).ravel()
            )
        )

        test_acc_hist.append(
            manual_accuracy(
                y_test,
                (y_hat_test >= 0.5).astype(int).ravel()
            )
        )

    return train_loss_hist, test_loss_hist, train_acc_hist, test_acc_hist


def draw_mlp_structure(input_nodes=8, hidden_nodes=8):
    plt.figure(figsize=(13, 7))

    input_x = 0
    hidden_x = 1.8
    output_x = 3.6

    input_y = np.linspace(0, 1, input_nodes)
    hidden_y = np.linspace(0.1, 0.9, hidden_nodes)
    output_y = [0.5]

    line_colors = [
        "#B388EB",
        "#F7AEF8",
        "#8093F1",
        "#72DDF7",
        "#F4A261",
        "#E76F51",
        "#2A9D8F",
        "#A7C957"
    ]

    for i, y1 in enumerate(input_y):
        for j, y2 in enumerate(hidden_y):
            plt.plot(
                [input_x, hidden_x],
                [y1, y2],
                color=line_colors[(i + j) % len(line_colors)],
                alpha=0.45,
                linewidth=1.2
            )

    for j, y2 in enumerate(hidden_y):
        plt.plot(
            [hidden_x, output_x],
            [y2, output_y[0]],
            color=line_colors[j % len(line_colors)],
            alpha=0.75,
            linewidth=1.4
        )

    plt.scatter(
        [input_x] * input_nodes,
        input_y,
        s=650,
        color="#6EC6FF",
        edgecolors="#1A5276",
        linewidths=1.5,
        zorder=3
    )

    plt.scatter(
        [hidden_x] * hidden_nodes,
        hidden_y,
        s=650,
        color="#B39DDB",
        edgecolors="#4A235A",
        linewidths=1.5,
        zorder=3
    )

    plt.scatter(
        [output_x],
        output_y,
        s=750,
        color="#FF8A80",
        edgecolors="#922B21",
        linewidths=1.5,
        zorder=3
    )

    plt.text(
        input_x,
        1.03,
        "Input Layer\n(30 features)",
        ha="center",
        fontsize=14,
        fontweight="bold"
    )

    plt.text(
        hidden_x,
        1.03,
        "Hidden Layer\n(15 nodes)",
        ha="center",
        fontsize=14,
        fontweight="bold"
    )

    plt.text(
        output_x,
        1.03,
        "Output Layer\n(1 node)",
        ha="center",
        fontsize=14,
        fontweight="bold"
    )

    plt.title(
        "MLP Structure for Breast Cancer Classification",
        fontsize=16,
        fontweight="bold",
        pad=35
    )

    plt.text(
        -0.25,
        -0.08,
        "Showing a subset of nodes for illustration",
        fontsize=11
    )

    plt.axis("off")
    plt.tight_layout()
    plt.savefig("figure1_mlp_structure.png", dpi=200, bbox_inches="tight")
    plt.show()


X, y = load_data("wdbc.data")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

X_train, X_test = standardize_manual(X_train, X_test)

np.random.seed(42)

epochs = 100

model = MLP(
    input_size=30,
    hidden_size=15,
    learning_rate=0.5
)

train_loss, test_loss, train_acc, test_acc = train_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    epochs
)

best_epoch = int(np.argmin(test_loss)) + 1

draw_mlp_structure()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(
    train_loss,
    color="#7B2CBF",
    linewidth=2.5,
    label="Train Loss"
)

axes[0].plot(
    test_loss,
    color="#FF006E",
    linewidth=2.5,
    label="Test Loss"
)

axes[0].axvline(
    x=best_epoch - 1,
    color="#2B2D42",
    linestyle="--",
    linewidth=2,
    label=f"Best Epoch = {best_epoch}"
)

axes[0].set_title("Model Loss", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Epochs")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True, linestyle="--", alpha=0.45)

axes[1].plot(
    train_acc,
    color="#00B4D8",
    linewidth=2.5,
    label="Train Accuracy"
)

axes[1].plot(
    test_acc,
    color="#FB8500",
    linewidth=2.5,
    label="Test Accuracy"
)

axes[1].axvline(
    x=best_epoch - 1,
    color="#2B2D42",
    linestyle="--",
    linewidth=2,
    label=f"Best Epoch = {best_epoch}"
)

axes[1].set_title("Model Accuracy", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Epochs")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(True, linestyle="--", alpha=0.45)

plt.suptitle(
    "Train/Test Loss and Accuracy (LR=0.5, Hidden=15)",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()
plt.savefig("figure2_loss_accuracy.png", dpi=200, bbox_inches="tight")
plt.show()

test_pred = model.predict(X_test)
print(f"Accuracy : {manual_accuracy(y_test, test_pred) * 100:.1f}%")
print(f"Precision: {manual_precision(y_test, test_pred) * 100:.1f}%")
print(f"Recall   : {manual_recall(y_test, test_pred) * 100:.1f}%")
print(f"F1-Score : {manual_f1(y_test, test_pred) * 100:.1f}%")

learning_rates = [1.0, 0.5, 0.1, 0.01]

plt.figure(figsize=(12, 6))

colors = {
    1.0: "#FF006E",
    0.5: "#8338EC",
    0.1: "#3A86FF",
    0.01:"#06D6A0"
}

for lr in learning_rates:
    np.random.seed(42)

    m = MLP(
        input_size=30,
        hidden_size=15,
        learning_rate=lr
    )

    trl, tel, _, _ = train_model(
        m,
        X_train,
        y_train,
        X_test,
        y_test,
        epochs
    )

    plt.plot(
        trl,
        color=colors[lr],
        linewidth=2,
        label=f"Train LR={lr}"
    )

    plt.plot(
        tel,
        color=colors[lr],
        linestyle="--",
        linewidth=2,
        label=f"Test LR={lr}"
    )

plt.title(
    "Loss vs Epochs for Different Learning Rates",
    fontsize=15,
    fontweight="bold"
)

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(fontsize=8)
plt.grid(True, linestyle="--", alpha=0.45)
plt.tight_layout()
plt.savefig("figure3_learning_rates.png", dpi=200, bbox_inches="tight")
plt.show()

hidden_sizes = [5, 10, 15, 20, 25, 30]
seeds = [1, 7, 21, 42, 99]

mean_accuracies = []

for k in hidden_sizes:
    accuracies_for_k = []

    for seed in seeds:
        np.random.seed(seed)

        m = MLP(
            input_size=30,
            hidden_size=k,
            learning_rate=0.5
        )

        _, _, _, test_acc_k = train_model(
            m,
            X_train,
            y_train,
            X_test,
            y_test,
            epochs
        )

        accuracies_for_k.append(test_acc_k[-1])

    mean_acc = np.mean(accuracies_for_k)
    mean_accuracies.append(mean_acc)

    print(f"k = {k}, Mean Test Accuracy = {mean_acc:.4f}")

plt.figure(figsize=(8, 5))

bars = plt.bar(
    [str(k) for k in hidden_sizes],
    mean_accuracies,
    color="#8E44AD",
    width=0.55
)

plt.xlabel("Number of Hidden Nodes (k)")
plt.ylabel("Average Test Classification Accuracy")
plt.title(
    "Accuracy vs. Number of Hidden Nodes (k)",
    fontsize=14,
    fontweight="bold"
)

plt.ylim(0.95, 1.00)

plt.yticks([
    0.95,
    0.96,
    0.97,
    0.98,
    0.99,
    1.00
])
plt.grid(axis="y", linestyle="--", alpha=0.4)

for bar, acc in zip(bars, mean_accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        acc + 0.001,
        f"{acc:.4f}",
        ha="center",
        fontsize=10,
        fontweight="bold"
    )

plt.tight_layout()
plt.savefig("figure4_hidden_nodes.png", dpi=200, bbox_inches="tight")
plt.show()