import torch
import numpy as np
import matplotlib.pyplot as plt


def evaluate_found_path(answer: torch.Tensor, Y_test: torch.Tensor):
    start = find_gate(answer, 0.6, True)
    end = find_gate(answer, 0.6, False)

    if start is None or end is None:
        return False

    current = (start[0], start[1])
    visited = set()
    visited.add(current)

    max_moves = (
        answer.shape[0] * answer.shape[1]
    )  # Consider the grid size for max moves
    # max_moves = 17

    for i in range(max_moves):
        brightest_neighbour = find_brightest_neighbour(answer, current, visited, Y_test)
        if brightest_neighbour is None:
            return False
        current = (brightest_neighbour[0], brightest_neighbour[1])
        visited.add(current)
        if current == end:
            return True
    return False


def find_gate(answer: torch.Tensor, epsillon: float, start: bool):
    rows, cols = answer.shape  # Get the actual number of rows and columns
    for row in range(rows):
        if start:
            if answer[row][0] > (3 - epsillon):
                return row, 0
        else:
            if answer[row][cols - 1] > (
                3 - epsillon
            ):  # Use the last column dynamically
                return row, cols - 1
    return None


def find_brightest_neighbour(
    answer: torch.Tensor,
    position: tuple[int, int],
    visited: set[tuple[int, int]],
    Y_test: torch.Tensor,
):
    rows, cols = answer.shape  # Get the actual number of rows and columns

    ind_up = (max(position[0] - 1, 0), position[1])
    ind_down = (min(position[0] + 1, rows - 1), position[1])
    ind_left = (position[0], max(position[1] - 1, 0))
    ind_right = (position[0], min(position[1] + 1, cols - 1))

    value_up = answer[ind_up[0]][ind_up[1]] if ind_up not in visited else -1
    value_down = answer[ind_down[0]][ind_down[1]] if ind_down not in visited else -1
    value_left = answer[ind_left[0]][ind_left[1]] if ind_left not in visited else -1
    value_right = answer[ind_right[0]][ind_right[1]] if ind_right not in visited else -1

    # Find the maximum value among the neighbours and return the corresponding index
    max_value = max(value_up, value_down, value_left, value_right)
    if max_value == -1:
        return None

    if max_value == value_up and Y_test[ind_up[0]][ind_up[1]] != 1.0:
        return ind_up
    elif max_value == value_down and Y_test[ind_down[0]][ind_down[1]] != 1.0:
        return ind_down
    elif max_value == value_left and Y_test[ind_left[0]][ind_left[1]] != 1.0:
        return ind_left
    elif max_value == value_right and Y_test[ind_right[0]][ind_right[1]] != 1.0:
        return ind_right

    return None


def evaluate_total_predictions_set(
    Y_hat: torch.Tensor, Y_test: torch.Tensor, verbose=True
):
    preds_: torch.Tensor = torch.zeros(Y_hat.shape[0], dtype=bool)

    # Dynamically determine the maze size from the data
    rows, cols = (
        Y_hat.shape[2],
        Y_hat.shape[3],
    )  # Assumes Y_hat has shape (batch_size, channels, rows, cols)

    wrong_predictions = []  # List to store images of wrong predictions

    for index in range(preds_.shape[0]):
        preds_[index] = evaluate_found_path(
            Y_hat[index].reshape(rows, cols), Y_test[index].numpy().reshape(rows, cols)
        )
        # Collect wrong predictions
        if not preds_[index]:
            wrong_predictions.append(Y_hat[index].reshape(rows, cols))

    total = preds_.shape[0]
    correct = preds_[preds_ == True].shape[0]
    correct_percentage = (correct / total) * 100

    if verbose:
        print("\n----------MODEL SUMMARY-----------")
        print("TOTAL SET SIZE: ", total)
        print("CORRECT GUESSES: ", correct)
        print("TOTALING TO ACCURACY%: ", correct_percentage)
        print("------------------------------------\n\n")

    if wrong_predictions:
        wrong_predictions_tensor = torch.stack(
            wrong_predictions
        )  # Stack wrong predictions into a tensor
    else:
        wrong_predictions_tensor = torch.empty(
            0
        )  # Return an empty tensor if no wrong predictions

    return correct_percentage, preds_, wrong_predictions_tensor


def display_mazes(
    mazes: torch.Tensor,
    n_rows: int = 1,
    n_cols: int = 5,
    figsize: tuple = (15, 3),
    max_mazes: int = 30,
):
    """
    Display mazes in a grid of n_rows x n_cols.
    Args:
        mazes: Tensor of mazes (batch_size, channels, rows, cols)
        n_rows: Number of rows in the grid (default 1)
        n_cols: Number of columns in the grid (default 5)
        figsize: Tuple specifying the size of the figure (default (15, 3))
    """
    n_rows = mazes.shape[0] // n_cols

    # Calculate total number of mazes to display
    num_mazes = min(len(mazes), n_rows * n_cols, max_mazes)

    n_rows = int(np.ceil(num_mazes / n_cols))
    figsize = (15, 3 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()  # Flatten the axes array to easily index them

    for i in range(n_rows * n_cols):
        # Each maze is assumed to be grayscale, so select the channel dimension if present
        if i < num_mazes:
            maze = mazes[i, 0].detach().cpu().numpy()
            axes[i].imshow(maze, cmap="cividis", interpolation="none")
            axes[i].set_title(f"Maze {i+1}")
        else:
            # Hide the unused subplot axes
            axes[i].axis("off")

        axes[i].axis("off")  # Hide axis labels and ticks

    plt.tight_layout()
    plt.show()

    return fig
