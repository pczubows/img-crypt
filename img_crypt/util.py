def pwlcm(x, p):
    """Piece wise linear chaotic map

    Parameters:
        x (float): Map argument
        p (float): Map control parameter. Should be in range from 0 to 0.5

    Return:
        float: Map return value
    """
    if 0 <= x < p:
        return x / p
    if p <= x < 0.5:
        return (x - p) / (0.5 - p)
    if 0.5 <= x < 1 - p:
        return (1 - x - p) / (0.5 - p)
    if 1 - p <= x <= 1:
        return (1 - x) / p
