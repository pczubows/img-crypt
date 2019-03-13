"""Speed Test

Script encrypts and decrypts sample image and prints
time it took for each scheme available.
"""

import timeit
from imgcrypt import encrypt_then_decrypt, example_urls


schemes = ['CML', 'CML_MULTI_PROC', '3DES_ECB', 'AES_ECB', '3DES_CBC', 'AES_ECB']

for scheme in schemes:
    start = timeit.default_timer()
    encrypt_then_decrypt(example_urls['obelix'], scheme=scheme, show=True)
    elapsed = timeit.default_timer() - start

    print(f'{scheme}  = {elapsed}')
