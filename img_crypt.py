import os
from io import BytesIO
from os import urandom
from pathlib import Path
from urllib.request import urlopen, pathname2url

from PIL import Image
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend


def make_bitmap(path, bw=False):
    im = image_from_url(path)

    path = Path(path)
    bmap_path = path.stem + '_bmap.bmp'

    if bw:
        im = im.convert('L')

    im.save(bmap_path)

    return bmap_path


def triple_des_encrypt(im, key):
    cipher = Cipher(algorithms.TripleDES(key), modes.ECB(), backend=default_backend())
    encryptor = cipher.encryptor()
    ct = encryptor.update(im.tobytes()) + encryptor.finalize()

    return Image.frombytes('RGB', im.size, ct)


def triple_des_decrypt(im, key):
    cipher = Cipher(algorithms.TripleDES(key), modes.ECB(), backend=default_backend())
    decryptor = cipher.decryptor()
    pt = decryptor.update(im.tobytes()) + decryptor.finalize()

    return Image.frombytes('RGB', im.size, pt)


def image_from_url(url):
    if os.path.isfile(url):
        url = 'file:' + pathname2url(url)

    with urlopen(url) as f:
        img_file = BytesIO(f.read())

    return Image.open(img_file)


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


def test_encryption(img_url, scheme='3DES', clean=False, show=True):
    encryption_schemes = {'3DES': {'encrypt': triple_des_encrypt, 'decrypt': triple_des_decrypt, 'key_size': 16}}

    path = Path(img_url)
    bmap_path = path.stem + '_bmap.bmp'

    if not os.path.isfile(bmap_path):
        make_bitmap(img_url)

    im = Image.open(bmap_path)

    encrypt = encryption_schemes[scheme]['encrypt']
    decrypt = encryption_schemes[scheme]['decrypt']
    key_len = encryption_schemes[scheme]['key_size']

    key = urandom(key_len)

    enc_im = encrypt(im, key)

    dec_im = decrypt(enc_im, key)

    if show:
        im.show()
        enc_im.show()
        dec_im.show()

    if clean:
        if os.path.isfile(bmap_path):
            os.remove(bmap_path)


if __name__ == '__main__':
    art_piece_url = 'https://scontent-frx5-1.xx.fbcdn.net/v/t1.15752-9/48381993_2910354935642160_243018625521287168_n.jpg?_nc_cat=103&_nc_ht=scontent-frx5-1.xx&oh=aff0bf4efe83872371eab8e5b122846f&oe=5CA0A95E'
    pit_url = 'https://m.media-amazon.com/images/M/MV5BMjE2OTIwMzE0MF5BMl5BanBnXkFtZTcwNjgyNjg0OQ@@._V1_.jpg'
    # im = image_from_url(pit_url)
    # im.show()

    test_encryption(art_piece_url)

