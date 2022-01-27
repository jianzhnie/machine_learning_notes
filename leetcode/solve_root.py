def f0(x, a=2):
    return x**2 - a


def f(x, a=2):
    return (x**2 - a)**2


def h(x, a=2):
    return 4 * x * (x**2 - a)


def fmin(x0, a=4, step=0.001, eta=1e-6):
    x = x0
    count = 0
    while f(x, a) > 0.00001:
        x = x - step * h(x, a)
        count += 1
    return x, count


def bainary_search(x0, a=4, eta=1e-6):
    left = 0.0
    right = a
    count = 0
    while abs(left - right) > eta:
        mid = (left + right) / 2
        count += 1
        if f0(mid, a) > 0:
            right = mid
        else:
            left = mid
    return mid, count


if __name__ == '__main__':
    x, count = fmin(1, a=4)
    print(x, count)
    print('fx', f(x))
    x, count = bainary_search(1, a=4)
    print(x, count)
