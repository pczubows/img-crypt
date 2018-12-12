from os import urandom
from pathlib import Path
from PIL import Image
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

def make_bitmap(path, bw=False):
    path = Path(path)
    bmap_path = path.stem + '_bmap.bmp'
    im = Image.open(path)

    if bw:
        im = im.convert('L')

    im.save(bmap_path)


def triple_DES_encrypt(im, key):
    cipher = Cipher(algorithms.TripleDES(key), modes.ECB(), backend=default_backend())
    encryptor = cipher.encryptor()
    ct = encryptor.update(im.tobytes()) + encryptor.finalize()

    return Image.frombytes('RGB', im.size, ct)

def triple_DES_decrypt(im, key):
    cipher = Cipher(algorithms.TripleDES(key), modes.ECB(), backend=default_backend())
    decryptor = cipher.decryptor()
    pt = decryptor.update(im.tobytes()) + decryptor.finalize()

    return Image.frombytes('RGB', im.size, pt)




"""
def convert_to_bin_array(im):
    def to_bit(data, i):
        base = int(i/8)
        shift = i % 8
        return (data[base] & (1 << shift)) >> shift

    data_bytes = io.BytesIO()
    im.save(data_bytes, format='BMP')
    data_bytes = data_bytes.getvalue()
    return [to_bit(data_bytes, i) for i in range(len(data_bytes) * 8)]
"""

"""
def convert_from_bin_array(bin_array, size):
    data_bytes = []
    bits_num = len(bin_array)
    bin_array = [str(i) for i in bin_array]
    i = 0

    while i < bits_num:
        data_bytes.append((int("0b" + "".join(bin_array[i:i + 8]), 2)))
        i += 8

    data_bytes = bytes(data_bytes)

    return Image.frombytes('RGB', size, data_bytes)
"""

if __name__ == '__main__':
    im = Image.open('lucjan_bmap.bmp')
    im.show(title='Odamawiam')

    triple_des_key = urandom(16)

    enc_im = triple_DES_encrypt(im, triple_des_key)
    enc_im.show()

    dec_im = triple_DES_decrypt(enc_im, triple_des_key)
    dec_im.show()


