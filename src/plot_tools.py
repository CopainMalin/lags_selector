from numpy import arange, zeros
from pandas import DataFrame
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from src.subsets import Subsets


def plot_scoring(
    sets: Subsets,
    preds: DataFrame,
    score: float,
    score_name: float,
    show_train: bool = True,
    save_path: str = None,
) -> None:
    plt.figure(figsize=(15, 5))
    plt.title(
        f"{score_name} on the test set : {score:.2f}",
        fontsize=12,
        fontweight="bold",
    )
    if show_train:
        plt.plot(
            sets.train.ds,
            sets.train.y,
            label="y_train",
            color="dodgerblue",
        )
    plt.plot(preds.ds, preds.iloc[:, -1], label="y_pred", color="#fb8500")
    plt.plot(
        sets.test.ds,
        sets.test.y,
        label="y_test",
        color="#023047",
    )
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()


def plot_rfe_score(
    prediction_results: DataFrame = None,
    error_results: DataFrame = None,
    save_path: str = None,
) -> None:
    plt.figure(figsize=(15, 5))
    if prediction_results is not None:
        plt.plot(prediction_results, label="MAE - prediction RFE", color="#fb8500")
        plt.scatter(
            x=prediction_results.idxmin(),
            y=prediction_results.min(),
            s=150,
            linewidth=2,
            edgecolors="#fb8500",
            facecolor="none",
            label=f"MAE : {prediction_results.min():.2f}",
        )
    if error_results is not None:
        plt.plot(error_results, color="#023047", label="MAE - error RFE")
        plt.scatter(
            x=error_results.idxmin(),
            y=error_results.min(),
            s=150,
            linewidth=2,
            edgecolors="#023047",
            facecolor="none",
            label=f"MAE : {error_results.min():.2f}",
        )
    plt.legend()
    plt.xticks(
        ticks=arange(len(prediction_results)),
        labels=arange(len(prediction_results), 0, -1),
    )
    plt.xlabel("Iteration on the recursive feature elimination process")
    plt.ylabel("MAE on the validation set")
    plt.title(
        "MAE evolution on the validation set during the recursive feature elimination process",
        fontweight="bold",
    )
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()


def plot_paths(
    prediction_results: DataFrame = None,
    error_results: DataFrame = None,
    save_path: str = None,
) -> None:
    grid_size = 24
    plt.figure(figsize=(15, 10))
    grid = zeros((grid_size, grid_size))

    path1 = [
        (x, y)
        for (x, y) in zip(
            prediction_results["lag_drop"] - 1,
            arange(prediction_results.shape[0]),
        )
    ]
    path2 = [
        (x, y)
        for (x, y) in zip(
            error_results["lag_drop"] - 1,
            arange(prediction_results.shape[0]),
        )
    ]

    for x, y in path1:
        grid[x, y] = 1
    for x, y in path2:
        grid[x, y] = 2

    cmap = ListedColormap(["white", "#023047", "#fb8500"])
    im = plt.imshow(grid, cmap=cmap, interpolation="nearest", origin="lower")
    plt.grid(color="#22223b")

    plt.xticks(arange(0.5, grid_size, 1.0), arange(1, grid_size + 1, 1))
    plt.xlabel("Dropped Lag")
    plt.ylabel("Iteration")
    plt.title(
        "Dropped lag per iteration in the recursive elimination process",
        fontweight="bold",
    )
    plt.yticks(arange(0.5, grid_size, 1.0), arange(1, grid_size + 1, 1))

    cax = plt.axes([0.78, 0.7, 0.02, 0.1])
    cbar = plt.colorbar(im, cax=cax, ticks=[1.7, 1])
    cbar.set_ticklabels(["Prediction contribution", "Error contribution"])

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()
