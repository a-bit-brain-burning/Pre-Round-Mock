practice-2026-11-16 - Problem A

| Contest base points | Maximum points | Official baseline total (T_std) |
| :--- | :--- | :--- |
| **1000** | **1500** | **201.978 ms** |

Points = base × min(1.5, T_std / total GPU time), rounded to 4 decimal places. Every submission is scored under the current T_std, so a T_std change rescores them all.

[Medium] [My submissions] [Leaderboard] [Rules]

Given a weighted directed graph represented as an adjacency matrix, compute the shortest distances between all pairs of vertices using the Floyd-Warshall algorithm.

### Input
*   `adj_matrix` : A 2D tensor of shape `(n, n)` representing the weighted adjacency matrix
    *   `adj_matrix[i][j]` = weight of edge from vertex `i` to vertex `j`
    *   `adj_matrix[i][j]` = `0` if there is no direct edge (except diagonal which represents self-loops)
    *   All weights are positive integers

### Output
*   A 2D tensor of shape `(n, n)` containing the shortest distances between all pairs of vertices
*   `output[i][j]` = shortest distance from vertex `i` to vertex `j`
*   If no path exists between vertices `i` and `j`, the distance should be `-1`

### Test Case Sizes
*   `n = 512`
*   `n = 1024`
*   `n = 2048`
*   `n = 4096`
