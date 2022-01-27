"""
input N * N
 transpose
"""


def transpose(M):
    N = 3
    for i in range(N):
        for j in range(i + 1, N):
            M[i, j] += M[j, i]
            M[j, i] = M[i, j] - M[j, i]
            M[i, j] = M[i, j] - M[j, i]
    return M
