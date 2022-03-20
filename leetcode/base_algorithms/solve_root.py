def fx(x, a):
    return (x**2 - a)**2


def gx(x, a):
    return 4 * x * (x**2 - a)


def solve_root(a, eps=1e-5, step=1e-3):
    x = float(a / 2)
    while fx(x, a) > eps:
        g = gx(x, a)
        x = x - step * g
    return x


# 梯度下降法求根
if __name__ == '__main__':
    nums = list(range(2, 20))
    for a in nums:
        root = solve_root(a)
        print(a, root)
