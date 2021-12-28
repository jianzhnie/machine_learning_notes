def f(x):
    return (x**2 - 2)**2


def h(x):
    return 4 * x * (x**2 - 2)


def fmin(x0, step=0.001):
    x = x0
    count = 0
    while (abs(f(x)) > 0.00001):
        x = x - step * h(x)
        count += 1
    return x, count


if __name__ == '__main__':
    x, count = fmin(1)
    print(x, count)
    print('fx', f(x))
