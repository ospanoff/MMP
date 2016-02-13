def naive(px, py):
    n = len(px)
    sum = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            sum += ((px[i] - px[j]) ** 2 + (py[i] - py[j]) ** 2) ** 0.5

    return 2 * sum / (n * (n - 1))
