import os

from io import BytesIO
from os import urandom
from random import seed, random
from pathlib import Path
from urllib.request import urlopen, pathname2url
from threading import Thread

from PIL import Image
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

import numpy as np


def make_bitmap(path, bw=False):
    im = image_from_url(path)

    path = Path(path)
    bmap_path = path.stem + '_bmap.bmp'

    if bw:
        im = im.convert('L')

    im.save(bmap_path)

    return bmap_path


def pwlcm(x, p):
    if 0 <= x < p:
        return x/p
    if p <= x < 0.5:
        return (x - p)/(0.5 - p)
    if 0.5 <= x < 1 - p:
        return (1 - x - p)/(0.5 - p)
    if 1 - p <= x <= 1:
        return (1 - x)/p


def cml_encrypt_channel(channel, p, iterations, cycles, parallel=False):
    pixels_flat = channel.flat
    pixel_num = len(pixels_flat)
    rand = [random() for _ in range(cycles * pixel_num)]

    for cycle in range(1, cycles + 1):
        for pixel_index in range(pixel_num):

            pixel_float = pixels_flat[pixel_index - 1] / 255

            for _ in range(iterations):
                pixel_float = pwlcm(pixel_float, p)

            k = pixel_num * (cycle - 1) + pixel_index
            pixel_float = (pixel_float + rand[k]) % 1
            pixels_flat[pixel_index] = pixels_flat[pixel_index] + round(pixel_float * 255)

            if pixels_flat[pixel_index] > 255:
                pixels_flat[pixel_index] = pixels_flat[pixel_index] - 256


def cml_decrypt_channel(channel, p, iterations, cycles, parallel=False):
    pixels_flat = channel.flat
    pixel_num = len(pixels_flat)
    rand = [random() for _ in range(cycles * pixel_num)]

    for cycle in range(cycles, 0, -1):
        for pixel_index in range(pixel_num - 1, -1, -1):

            pixel_float = pixels_flat[pixel_index - 1] / 255

            for _ in range(iterations):
                pixel_float = pwlcm(pixel_float, p)

            k = pixel_num * (cycle - 1) + pixel_index
            pixel_float = (pixel_float + rand[k]) % 1
            pixels_flat[pixel_index] = pixels_flat[pixel_index] - round(pixel_float * 255)

            if pixels_flat[pixel_index] < 0:
                pixels_flat[pixel_index] = pixels_flat[pixel_index] + 256


def cml_encrypt(im, key, iterations=25, cycles=5):
    pixels = np.array(im)
    p, s = key
    seed(s)

    for i in range(3):
        cml_encrypt_channel(pixels[:, :, i], p, iterations, cycles)

    return Image.fromarray(pixels)


def cml_decrypt(im, key, iterations=25, cycles=5):
    pixels = np.array(im)
    p, s = key
    seed(s)

    for i in range(3):
        cml_decrypt_channel(pixels[:, :, i], p, iterations, cycles)

    return Image.fromarray(pixels)


def cml_para_encrypt(im, key, iterations=25, cycles=5):
    pixels = np.array(im)
    p, s = key
    seed(s)
    threads = []

    for i in range(3):
        t = Thread(target=cml_encrypt_channel, args=(pixels[:, :, i], p, iterations, cycles))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    return Image.fromarray(pixels)


def cml_para_decrypt(im, key, iterations=25, cycles=5):
    pixels = np.array(im)
    p, s = key
    seed(s)
    threads = []

    for i in range(3):
        t = Thread(target=cml_decrypt_channel, args=(pixels[:, :, i], p, iterations, cycles))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    return Image.fromarray(pixels)


def standard_encrypt(im, algorithm, mode):
    data = im.tobytes()
    cipher = Cipher(algorithm, mode, backend=default_backend())
    encryptor = cipher.encryptor()
    padder = padding.PKCS7(algorithm.block_size).padder()

    padded_data = padder.update(data) + padder.finalize()
    ciphertext = encryptor.update(padded_data) + encryptor.finalize()

    return Image.frombytes('RGB', im.size, ciphertext)


def standard_decrypt(im, algorithm, mode):
    data = im.tobytes()
    cipher = Cipher(algorithm, mode, backend=default_backend())
    decryptor = cipher.decryptor()
    padder = padding.PKCS7(algorithm.block_size).padder()

    padded_ciphertext = padder.update(data) + padder.finalize()
    plaintext = decryptor.update(padded_ciphertext) + decryptor.finalize()

    return Image.frombytes('RGB', im.size, plaintext)


def image_from_url(url):
    if os.path.isfile(url):
        url = 'file:' + pathname2url(url)

    with urlopen(url) as f:
        img_file = BytesIO(f.read())

    return Image.open(img_file)


def test_encryption(img_url, scheme='3DES_ECB', clean=False, show=True, **kwargs):
    path = Path(img_url)
    bmap_path = path.stem + '_bmap.bmp'

    if path.suffix == 'bmp' and os.path.isfile(path):
        bmap_path = path
    elif not os.path.isfile(bmap_path):
        make_bitmap(img_url)

    im = Image.open(bmap_path)

    if scheme == '3DES_ECB':
        key = urandom(16)
        enc_im = standard_encrypt(im, algorithms.TripleDES(key), modes.ECB())
        dec_im = standard_decrypt(enc_im, algorithms.TripleDES(key), modes.ECB())
    elif scheme == '3DES_CBC':
        key = urandom(16)
        iv = urandom(8)
        enc_im = standard_encrypt(im, algorithms.TripleDES(key), modes.CBC(iv))
        dec_im = standard_decrypt(enc_im, algorithms.TripleDES(key), modes.CBC(iv))
    elif scheme == 'AES_ECB':
        key = urandom(16)
        enc_im = standard_encrypt(im, algorithms.AES(key), modes.ECB())
        dec_im = standard_decrypt(enc_im, algorithms.AES(key), modes.ECB())
    elif scheme == 'AES_CBC':
        key = urandom(16)
        iv = urandom(16)
        enc_im = standard_encrypt(im, algorithms.AES(key), modes.CBC(iv))
        dec_im = standard_decrypt(enc_im, algorithms.AES(key), modes.CBC(iv))
    elif scheme == 'CML' or 'CML_PARA':
        if kwargs.get('key') is not None:
            key = kwargs['key']
        else:
            key = 0.3, urandom(4)
        if kwargs.get('cycles') is not None:
            cycles = kwargs['cycles']
        else:
            cycles = 5
        if kwargs.get('iterations') is not None:
            iterations = kwargs['iterations']
        else:
            iterations = 25

        if scheme == 'CML':
            enc_im = cml_encrypt(im, key, iterations=iterations, cycles=cycles)
            dec_im = cml_decrypt(enc_im, key, iterations=iterations, cycles=cycles)
        else:
            enc_im = cml_para_encrypt(im, key, iterations=iterations, cycles=cycles)
            dec_im = cml_para_decrypt(enc_im, key, iterations=iterations, cycles=cycles)
    else:
        raise Exception('Invalid encryption scheme')

    if show:
        im.show()
        enc_im.show()
        dec_im.show()

    if clean:
        if os.path.isfile(bmap_path):
            os.remove(bmap_path)


if __name__ == '__main__':
    url_dict = {'art_piece': 'https://scontent-frx5-1.xx.fbcdn.net/v/t1.15752-9'
                             '/48381993_2910354935642160_243018625521287168_n.jpg?_nc_cat=103&_nc_ht=scontent'
                             '-frx5-1.xx&oh=aff0bf4efe83872371eab8e5b122846f&oe=5CA0A95E',
                'pit': 'https://m.media-amazon.com/images/M/MV5BMjE2OTIwMzE0MF5BMl5BanBnXkFtZTcwNjgyNjg0OQ@@._V1_'
                       '.jpg',
                'hacker': 'https://pbs.twimg.com/media/CnwWR8HXgAAToWA.jpg',
                'bog': 'https://i.kym-cdn.com/photos/images/original/001/396/633/d76.jpg'}

    test_encryption(url_dict['bog'], scheme='CML_PARA', iterations=10, cycles=5)
