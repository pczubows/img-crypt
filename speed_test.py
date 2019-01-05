import timeit

from img_crypt import test_encryption


schemes = {'CML', 'CML_MULTI_PROC'}
url = 'img/small.jpeg'

for scheme in schemes:
    start = timeit.default_timer()
    test_encryption(url, scheme=scheme, show=True)
    elapsed = timeit.default_timer() - start

    print(f'{scheme}  = {elapsed}')
