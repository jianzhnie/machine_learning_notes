def binary_search(a, eps=1e-3):
    left = 0
    right = a
    mid = (left + right) / 2
    if mid**2 - a > eps:
        right = mid
    elif mid**2 - a < eps:
        left = mid
    else:
        return mid
    return mid
