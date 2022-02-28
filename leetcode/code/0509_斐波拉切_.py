def fib(n):
    if n <= 1:
        return n
    else:
        return fib(n - 1) + fib(n - 2)


def fib2(n):
    nums = [0] * (n + 1)
    nums[1] = 1
    for i in range(2, n + 1):
        nums[i] = nums[i - 1] + nums[i - 2]
    return nums[n]


def fib3(n):
    a = 0
    b = 1
    if n < 2:
        return n
    for i in range(2, n + 1):
        c = a + b
        a = b
        b = c
    return c
