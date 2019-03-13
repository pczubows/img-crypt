def pwlcm(x, p):
    if 0 <= x < p:
        return x / p
    if p <= x < 0.5:
        return (x - p) / (0.5 - p)
    if 0.5 <= x < 1 - p:
        return (1 - x - p) / (0.5 - p)
    if 1 - p <= x <= 1:
        return (1 - x) / p
