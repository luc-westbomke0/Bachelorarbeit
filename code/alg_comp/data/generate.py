import numpy as np
import random
import multiprocessing
import time
import glob
import os


def generate_maze_aldous_broder(size):
    rows, cols = size

    maze = np.ones((rows, cols))

    visited = np.zeros((rows, cols), dtype=bool)

    def in_bounds(x, y):
        return 0 <= x < rows and 0 <= y < cols

    def get_neighbors(x, y):
        neighbors = []
        for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            nx, ny = x + dx, y + dy
            if in_bounds(nx, ny):
                neighbors.append((nx, ny))
        return neighbors

    start_x, start_y = random.randrange(0, rows, 2), random.randrange(0, cols, 2)
    maze[start_x, start_y] = 0
    visited[start_x, start_y] = True

    total_cells = int(np.ceil(rows / 2)) * int(np.ceil(cols / 2))
    visited_count = 1
    x, y = start_x, start_y

    while visited_count < total_cells:
        neighbors = get_neighbors(x, y)
        if len(neighbors) == 0:
            break
        nx, ny = random.choice(neighbors)

        if not visited[nx, ny]:
            mx, my = (x + nx) // 2, (y + ny) // 2
            maze[mx, my] = 0
            maze[nx, ny] = 0
            visited[nx, ny] = True
            visited_count += 1

        x, y = nx, ny

    gate_count = 2
    min_distance = min(rows, cols) // 2

    gates = []
    for _ in range(gate_count):
        while True:
            gx, gy = random.randint(0, rows - 1), random.randint(0, cols - 1)
            if maze[gx, gy] == 0:
                if all(
                    abs(gx - gx2) + abs(gy - gy2) >= min_distance for gx2, gy2 in gates
                ):
                    maze[gx, gy] = 3
                    gates.append((gx, gy))
                    break

    return maze


def a_star_solve_no_obstacles(maze):
    rows, cols = maze.shape
    gates = np.argwhere(maze == 3)
    if len(gates) < 2:
        return maze

    start = tuple(gates[0])
    end = tuple(gates[1])

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    open_set = [start]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, end)}

    while open_set:
        current = min(open_set, key=lambda x: f_score.get(x, float("inf")))
        open_set.remove(current)

        if current == end:
            solved_maze = np.where(maze == 1, 0, maze)
            x, y = current
            while current in came_from:
                x, y = current
                if solved_maze[x, y] == 0:
                    solved_maze[x, y] = 2
                current = came_from[current]
            return solved_maze

        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)

            if (
                0 <= neighbor[0] < rows
                and 0 <= neighbor[1] < cols
                and maze[neighbor] != 1
            ):
                tentative_g_score = g_score[current] + 1

                if tentative_g_score < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                    if neighbor not in open_set:
                        open_set.append(neighbor)

    return maze


def worker_process(size, num_mazes, results):
    batch_unsolved = []
    batch_solved = []

    for i in range(num_mazes):
        maze = generate_maze_aldous_broder(size)
        solved_maze = a_star_solve_no_obstacles(maze.copy())
        batch_unsolved.append(maze)
        batch_solved.append(solved_maze)

    results.put((batch_unsolved, batch_solved))


def generate_and_solve_mazes_parallel(size, num_mazes, batch_size=10_000):
    start_time = time.time()
    num_processes = multiprocessing.cpu_count()
    manager = multiprocessing.Manager()
    results = manager.Queue()

    # Calculate total batches
    num_batches = (num_mazes + batch_size - 1) // batch_size

    for batch_index in range(num_batches):
        print(f"Processing batch {batch_index + 1}/{num_batches}...")

        mazes_in_batch = min(batch_size, num_mazes - batch_index * batch_size)
        mazes_per_process = mazes_in_batch // num_processes
        mazes_per_process = [mazes_per_process] * num_processes
        mazes_per_process[-1] += mazes_in_batch - sum(mazes_per_process)

        processes = []

        for i in range(num_processes):
            p = multiprocessing.Process(
                target=worker_process, args=(size, mazes_per_process[i], results)
            )
            processes.append(p)
            p.start()

        batch_unsolved = []
        batch_solved = []
        for _ in range(len(processes)):
            b_unsolved, b_solved = results.get()
            batch_unsolved.extend(b_unsolved)
            batch_solved.extend(b_solved)

        for p in processes:
            p.join()

        batch_unsolved = np.stack(batch_unsolved)
        batch_solved = np.stack(batch_solved)

        np.save(
            f"./batch_{batch_index + 1}_unsolved.npy",
            batch_unsolved[:, np.newaxis],
        )
        np.save(
            f"./batch_{batch_index + 1}_solved.npy",
            batch_solved[:, np.newaxis],
        )

        del batch_unsolved, batch_solved

    unsolved_files = sorted(glob.glob("./batch_*_unsolved.npy"))
    solved_files = sorted(glob.glob("./batch_*_solved.npy"))

    print("Combining all batches into single files...")
    all_unsolved = np.concatenate([np.load(f) for f in unsolved_files], axis=0)
    all_solved = np.concatenate([np.load(f) for f in solved_files], axis=0)

    unsolved_name = f"./{all_unsolved.shape[0]}x{all_unsolved.shape[2]}x{all_unsolved.shape[3]}_unsolved.npy"
    solved_name = f"./{all_unsolved.shape[0]}x{all_unsolved.shape[2]}x{all_unsolved.shape[3]}_solved.npy"

    np.save(
        unsolved_name,
        all_unsolved,
    )
    np.save(
        solved_name,
        all_solved,
    )

    for f in unsolved_files + solved_files:
        os.remove(f)

    elapsed_time = time.time() - start_time
    print(f"Processed {num_mazes} mazes in {elapsed_time:.2f}s.")
    print(f"All mazes processed and saved into '{unsolved_name}' and '{solved_name}'.")


if __name__ == "__main__":
    generate_and_solve_mazes_parallel(size=(10, 10), num_mazes=100_000)
