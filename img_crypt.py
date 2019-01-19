import os

from io import BytesIO
from os import urandom
from random import seed, random
from pathlib import Path
from urllib.request import urlopen, pathname2url
from threading import Thread, Event
from multiprocessing import Process, Queue

from PIL import Image
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

import numpy as np
import matplotlib.pyplot as plt


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


def cml_encrypt_channel(channel, p, iterations, cycles):
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


def cml_decrypt_channel(channel, p, iterations, cycles):
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


def cml_para_encrypt_channel(channel, p, iterations, cycles, prng_finished):
    pixels_flat = channel.flat
    pixel_num = len(pixels_flat)
    prng_finished.clear()
    rand = [random() for _ in range(cycles * pixel_num)]
    prng_finished.set()

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


def cml_para_decrypt_channel(channel, p, iterations, cycles, prng_finished):
    pixels_flat = channel.flat
    pixel_num = len(pixels_flat)
    prng_finished.clear()
    rand = [random() for _ in range(cycles * pixel_num)]
    prng_finished.set()

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


def cml_para_thread_encrypt(im, key, iterations=25, cycles=5):
    pixels = np.array(im)
    p, s = key
    seed(s)
    threads = []
    prng_finished = Event()
    prng_finished.set()

    for i in range(3):
        t = Thread(target=cml_para_encrypt_channel, args=(pixels[:, :, i], p, iterations, cycles, prng_finished))
        threads.append(t)
        prng_finished.wait()
        t.start()

    for t in threads:
        t.join()

    return Image.fromarray(pixels)


def cml_para_thread_decrypt(im, key, iterations=25, cycles=5):
    pixels = np.array(im)
    p, s = key
    seed(s)
    threads = []
    prng_finished = Event()
    prng_finished.set()

    for i in range(3):
        t = Thread(target=cml_para_decrypt_channel, args=(pixels[:, :, i], p, iterations, cycles, prng_finished))
        threads.append(t)
        prng_finished.wait()
        t.start()

    for t in threads:
        t.join()

    return Image.fromarray(pixels)


class MultiprocHandler:
    def __init__(self):
        self.queues = [Queue() for _ in range(3)]

    def cml_proc_encrypt_channel(self, channel_index, channel, rand, p, iterations, cycles):
        pixels_flat = channel.flat
        pixel_num = len(pixels_flat)

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

        self.queues[channel_index].put(channel)

    def cml_proc_decrypt_channel(self, channel_index, channel, rand, p, iterations, cycles):
        pixels_flat = channel.flat
        pixel_num = len(pixels_flat)

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

        self.queues[channel_index].put(channel)

    def cml_para_proc_encrypt(self, im, key, iterations=25, cycles=5):
        pixels = np.array(im)
        channel_size = pixels[:, :, 0].size

        dims = ()
        x, y = im.size
        dims += (y, x)
        dims += (3,)

        new_pixels = np.empty(dims, dtype=np.uint8)

        p, s = key
        seed(s)

        rand = []
        for i in range(3):
            rand.append([random() for _ in range(cycles * channel_size)])

        for i in range(3):
            proc = Process(target=self.cml_proc_encrypt_channel, args=(i, pixels[:, :, i], rand[i], p, iterations, cycles))
            proc.start()

        for i in range(3):
            new_pixels[:, :, i] = self.queues[i].get()

        return Image.fromarray(new_pixels)

    def cml_para_proc_decrypt(self, im, key, iterations=25, cycles=5):
        pixels = np.array(im)
        channel_size = pixels[:, :, 0].size

        dims = ()
        x, y = im.size
        dims += (y, x)
        dims += (3,)

        new_pixels = np.empty(dims, dtype=np.uint8)

        p, s = key
        seed(s)

        rand = []

        for i in range(3):
            rand.append([random() for _ in range(cycles * channel_size)])

        for i in range(3):
            proc = Process(target=self.cml_proc_decrypt_channel, args=(i, pixels[:, :, i], rand[i], p, iterations, cycles))
            proc.start()

        for i in range(3):
            new_pixels[:, :, i] = self.queues[i].get()

        return Image.fromarray(new_pixels)


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


def test_encryption(img_url, scheme='3DES_ECB', clean=False, show=True, hists=False, **kwargs):
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
    elif scheme == 'CML' or 'CML_MULTI_THREAD' or 'CML_MULTI_PROC':
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
        elif scheme == 'CML_MULTI_THREAD':
            enc_im = cml_para_thread_encrypt(im, key, iterations=iterations, cycles=cycles)
            dec_im = cml_para_thread_decrypt(enc_im, key, iterations=iterations, cycles=cycles)
        else:
            handler = MultiprocHandler()
            enc_im = handler.cml_para_proc_encrypt(im, key, iterations=iterations, cycles=cycles)
            dec_im = handler.cml_para_proc_decrypt(enc_im, key, iterations=iterations, cycles=cycles)
    else:
        raise Exception('Invalid encryption scheme')

    if show:
        im.show()
        enc_im.show()
        dec_im.show()
    if hists:
        plot_channels(im, f'{path.stem} plain')
        plot_channels(enc_im, f'{path.stem} {scheme.replace("_", " ").lower()} encrypted')
    if clean:
        if os.path.isfile(bmap_path):
            os.remove(bmap_path)


def encrypt_to_file(im, path, scheme, **kwargs):
    path = Path(path)

    if path.suffix != '.bmp':
        raise Exception('zapis tylko do bitmapy')

    key = kwargs.get('key')
    iv = kwargs.get('iv')
    iterations = kwargs.get('iterations')
    cycles = kwargs.get('cycles')

    if scheme == '3DES_ECB':
        enc_im = standard_encrypt(im, algorithms.TripleDES(key), modes.ECB())
    elif scheme == '3DES_CBC':
        enc_im = standard_encrypt(im, algorithms.TripleDES(key), modes.CBC(iv))
    elif scheme == 'AES_ECB':
        enc_im = standard_encrypt(im, algorithms.AES(key), modes.ECB())
    elif scheme == 'AES_CBC':
        enc_im = standard_encrypt(im, algorithms.AES(key), modes.CBC(iv))
    elif scheme == 'CML' or 'CML_MULTI_THREAD' or 'CML_MULTI_PROC':
        if scheme == 'CML':
            enc_im = cml_encrypt(im, key, iterations=iterations, cycles=cycles)
        elif scheme == 'CML_MULTI_THREAD':
            enc_im = cml_para_thread_encrypt(im, key, iterations=iterations, cycles=cycles)
        else:
            handler = MultiprocHandler()
            enc_im = handler.cml_para_proc_encrypt(im, key, iterations=iterations, cycles=cycles)
    else:
        raise Exception('Invalid encryption scheme')

    enc_im.save(path)


def decrypt_file(path, scheme, **kwargs):
    path = Path(path)

    if path.suffix != '.bmp':
        raise Exception('wczytywanie tylko z bitmapy')

    im = Image.open(path)

    key = kwargs.get('key')
    iv = kwargs.get('iv')
    iterations = kwargs.get('iterations')
    cycles = kwargs.get('cycles')

    if scheme == '3DES_ECB':
        dec_im = standard_decrypt(im, algorithms.TripleDES(key), modes.ECB())
    elif scheme == '3DES_CBC':
        dec_im = standard_decrypt(im, algorithms.TripleDES(key), modes.CBC(iv))
    elif scheme == 'AES_ECB':
        dec_im = standard_decrypt(im, algorithms.AES(key), modes.ECB())
    elif scheme == 'AES_CBC':
        dec_im = standard_decrypt(im, algorithms.AES(key), modes.CBC(iv))
    elif scheme == 'CML' or 'CML_MULTI_THREAD' or 'CML_MULTI_PROC':
        if scheme == 'CML':
            dec_im = cml_decrypt(im, key, iterations=iterations, cycles=cycles)
        elif scheme == 'CML_MULTI_THREAD':
            dec_im = cml_para_thread_decrypt(im, key, iterations=iterations, cycles=cycles)
        else:
            handler = MultiprocHandler()
            dec_im = handler.cml_para_proc_decrypt(im, key, iterations=iterations, cycles=cycles)
    else:
        raise Exception('Invalid decryption scheme')

    return dec_im


def plot_channels(im, title):
    pixels = np.array(im)
    channels = [pixels[:, :, i].ravel() for i in range(3)]

    plt.hist(channels, 256, [0, 256], color=['red', 'green', 'blue'])
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    url_dict = {'art_piece': 'https://scontent-frx5-1.xx.fbcdn.net/v/t1.15752-9'
                             '/48381993_2910354935642160_243018625521287168_n.jpg?_nc_cat=103&_nc_ht=scontent'
                             '-frx5-1.xx&oh=aff0bf4efe83872371eab8e5b122846f&oe=5CA0A95E',
                'pit': 'https://m.media-amazon.com/images/M/MV5BMjE2OTIwMzE0MF5BMl5BanBnXkFtZTcwNjgyNjg0OQ@@._V1_'
                       '.jpg',
                'hacker': 'https://pbs.twimg.com/media/CnwWR8HXgAAToWA.jpg',
                'bog': 'https://i.kym-cdn.com/photos/images/original/001/396/633/d76.jpg',
                'sultan': 'http://img1.garnek.pl/a.garnek.pl/004/866/4866080_800.0.jpg/sultan.jpg'}

    test_encryption(url_dict['sultan'], scheme='CML_MULTI_PROC', hists=True, iterations=50, cycles=10)
    test_encryption(url_dict['sultan'], scheme='3DES_CBC', hists=True)
