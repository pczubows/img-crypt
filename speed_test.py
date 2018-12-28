import timeit

from img_crypt import test_encryption


schemes = {'3DES_ECB', '3DES_CBC', 'AES_ECB', 'AES_CBC'}
url = 'img/cursed.png'

for scheme in schemes:
    start = timeit.default_timer()
    test_encryption(url, scheme=scheme, show=True)
    elapsed = timeit.default_timer() - start

    print(f'{scheme}  = {elapsed}')
