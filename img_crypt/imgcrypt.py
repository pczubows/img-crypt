"""Img Crypt

Script for testing different image encryption methods. It implements
traditional stream ciphers and an algorithm designed for image encryption
based on chaotic map lattices.

Chosen image is being encrypted and decrypted. When procedure finishes script
displays image before and after encryption, along with histograms of their respective
pixel values.

Usage is described in README.md
"""

import os

from argparse import ArgumentParser
from io import BytesIO
from urllib.request import urlopen, pathname2url, urlparse

from PIL import Image
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import numpy as np
import matplotlib.pyplot as plt

from multiproc_handler import MultiprocHandler
from cml import cml_encrypt, cml_decrypt
from exceptions import SchemeException, ImageLocatorException


def image_from_url(locator):
    """Load image from path or url

    Parameters:
        locator (str): Path to the image or url of an image hosted on the internet

    Returns:
        PIL.Image: Chosen image converted to the Pillow Image object
    """
    if os.path.isfile(locator):
        locator = 'file:' + pathname2url(locator)
    else:
        url_parsed = urlparse(locator)

        if url_parsed.netloc == '':
            raise ImageLocatorException(f'Wrong image locator {locator}, file does not exist or url is malformed')

    with urlopen(locator) as f:
        img_file = BytesIO(f.read())

    im = Image.open(img_file)

    if im.mode != 'RGB':
        im = im.convert('RGB')

    return im


def standard_encrypt(im, algorithm, mode):
    """Encrypt image using one the traditional stream ciphers

    Parameters:
        im (PIL.Image): Image to be encrypted loaded into Pillow Image object
        algorithm : Cryptography object with stream cipher algorithm
        mode: Cryptography object with encryption mode

    Returns:
        PIL.Image: Encrypted image object
    """
    data = im.tobytes()
    cipher = Cipher(algorithm, mode, backend=default_backend())
    encryptor = cipher.encryptor()
    padder = padding.PKCS7(algorithm.block_size).padder()

    padded_data = padder.update(data) + padder.finalize()
    ciphertext = encryptor.update(padded_data) + encryptor.finalize()

    return Image.frombytes('RGB', im.size, ciphertext)


def standard_decrypt(im, algorithm, mode):
    """Decrypt image using one the traditional stream ciphers

    Parameters:
        im (PIL.Image): Image to be encrypted loaded into Pillow Image object
        algorithm : Cryptography object with stream cipher algorithm
        mode: Cryptography object with encryption mode

    Returns:
        PIL.Image: Decrypted image object
    """
    data = im.tobytes()
    cipher = Cipher(algorithm, mode, backend=default_backend())
    decryptor = cipher.decryptor()
    padder = padding.PKCS7(algorithm.block_size).padder()

    padded_ciphertext = padder.update(data) + padder.finalize()
    plaintext = decryptor.update(padded_ciphertext) + decryptor.finalize()

    return Image.frombytes('RGB', im.size, plaintext)


def encrypt_image(im, scheme, **kwargs):
    """Encrypt image with chosen scheme

    Parameters:
        locator (str): Path or url of chosen image
        scheme (str): String specifying chosen encryption scheme

    Keyword Arguments:
        key (bytes | (float, bytes)): Encryption key, length and type depends on the chosen
            algorithm
        iv (bytes): Initialization vector for cbc encryption mode
        iterations (int): number of iterations over single pixel for CML algorithm
        cycles (int): number of cycles over whole image for cml algorithm

    Returns:
        PIL.Image: Encrypted image as Pillow Image object
    """
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
    elif scheme == 'CML':
        enc_im = cml_encrypt(im, key, iterations=iterations, cycles=cycles)
    elif scheme == 'CML_MULTI':
        handler = MultiprocHandler()
        enc_im = handler.multiproc_encrypt(im, key, iterations=iterations, cycles=cycles)
    else:
        raise SchemeException('Invalid encryption scheme')

    return enc_im


def decrypt_image(im, scheme, **kwargs):
    """Decrypt image with chosen scheme

    Parameters:
        locator (str): Path or url of chosen image
        scheme (str): String specifying chosen encryption scheme

    Keyword Arguments:
        key (bytes | (float, bytes)): Encryption key, length and type depends on the chosen
            algorithm
        iv (bytes): Initialization vector for cbc encryption mode
        iterations (int): number of iterations over single pixel for CML algorithm
        cycles (int): number of cycles over whole image for cml algorithm

    Returns:
        PIL.Image: Decrypted image as Pillow Image object
    """
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
    elif scheme == 'CML_MULTI':
        handler = MultiprocHandler()
        dec_im = handler.multiproc_decrypt(im, key, iterations=iterations, cycles=cycles)
    elif scheme == 'CML':
        dec_im = cml_decrypt(im, key, iterations=iterations, cycles=cycles)
    else:
        raise SchemeException('Invalid decryption scheme')

    return dec_im


def encrypt_then_decrypt(locator, scheme, **kwargs):
    """Perform encryption and decryption consecutively

    Parameters:
        locator (str): Path or url of chosen image
        scheme (str): String specifying chosen encryption scheme

    Keyword Arguments:
        key (bytes | (float, bytes)): Encryption key, length and type depends on the chosen
            algorithm
        iv (bytes): Initialization vector for cbc encryption mode
        iterations (int): number of iterations over single pixel for CML algorithm
        cycles (int): number of cycles over whole image for cml algorithm

    Returns:
        (PIL.Image, PIL.Image): Tuple containing decrypted image and encrypted image
    """
    im = image_from_url(locator)

    enc_im = None
    dec_im = None

    key = kwargs.get('key')
    iv = kwargs.get('iv')
    iterations = kwargs.get('iterations')
    cycles = kwargs.get('cycles')

    if scheme in ['3DES_ECB', 'AES_ECB', '3DES_CBC', 'AES_CBC']:
        key = key if key is not None else os.urandom(16)
        if scheme in ['3DES_ECB', 'AES_ECB']:
            enc_im = encrypt_image(im, scheme, key=key)
            dec_im = decrypt_image(enc_im, scheme, key=key)
        else:
            if scheme == '3DES_CBC':
                iv = iv if iv is not None else os.urandom(8)
            else:
                iv = iv if iv is not None else os.urandom(16)

            enc_im = encrypt_image(im, scheme, key=key, iv=iv)
            dec_im = decrypt_image(enc_im, scheme, key=key, iv=iv)

    elif scheme in ['CML', 'CML_MULTI']:
        cml_default_cycles = 3
        cml_default_iterations = 10

        key = key if key is not None else 0.3, os.urandom(4)
        cycles = cycles if cycles is not None else cml_default_cycles
        iterations = iterations if iterations is not None else cml_default_iterations

        enc_im = encrypt_image(im, scheme, key=key, iterations=iterations, cycles=cycles)
        dec_im = decrypt_image(enc_im, scheme, key=key, iterations=iterations, cycles=cycles)

    else:
        raise SchemeException('Invalid encryption scheme')

    return dec_im, enc_im


def plot_channels(dec_im, enc_im, scheme):
    """Draw histograms of the decrypted and encrypted image

    Parameters:
        dec_im (PIL.Image): Pillow Image object containing decrypted Image.
        enc_im (PIL.Image): Pillow Image object containing encrypted Image
        scheme (str): String specifying encryption scheme

    """
    dec_pixels = np.array(dec_im)
    enc_pixels = np.array(enc_im)

    dec_channels = [dec_pixels[:, :, i].ravel() for i in range(3)]
    enc_channels = [enc_pixels[:, :, i].ravel() for i in range(3)]

    hist_kwargs = {'bins': 256, 'range': [0, 256], 'color': ['red', 'green', 'blue']}

    fig, (hist_dec, hist_enc) = plt.subplots(nrows=1, ncols=2)
    hist_dec.hist(dec_channels, **hist_kwargs)
    hist_enc.hist(enc_channels, **hist_kwargs)
    hist_dec.set_title('plain')
    hist_enc.set_title(f"{scheme.replace('_', ' ').lower()} encrypted")
    plt.show()


# example pictures for tests
example_urls = {
    'obelix': "https://www.asterix.com/illus/asterix-de-a-a-z/les-personnages/perso/g28b.gif",
    'matterhorn': "https://www.zermatt.ch/extension/portal-zermatt/var/storage/images/media/bibliothek/berge/matterhorn/sicht-aufs-matterhorn-vom-gornergrat/58955-3-ger-DE/Sicht-aufs-Matterhorn-vom-Gornergrat_grid_624x350.jpg",
    'cat': "https://www.petmd.com/sites/default/files/what-does-it-mean-when-cat-wags-tail.jpg",
    'balloon': "https://i.pinimg.com/originals/33/c8/b0/33c8b01484886f1b075ee2d1d7b9d12b.gif",
}


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('-s', '--scheme', type=str, default='3DES_CBC', help='specify encryption scheme'
                                                             'allowed schemes: 3DES_ECB, 3DES_CBC, AES_ECB, AES_CBC, CML, CML_MULTI')
    arg_parser.add_argument('-l', '--locator', type=str, default=None,
                            help='specify image to be encrypted, can be path to the local'
                                 'file or url of an image hosted on internet')
    arg_parser.add_argument('-e', '--example', type=str, default=None,
                            help=f'Name of example picture to be fetched from internet'
                                 f'Example names {example_urls.keys()}')

    cl_args = arg_parser.parse_args()

    scheme = cl_args.scheme.upper()
    cli_locator = cl_args.locator
    example = cl_args.example

    if cli_locator is not None:
        locator = cli_locator
    elif example in example_urls.keys():
        locator = example_urls[example]
    else:
        raise ImageLocatorException('No proper image locator specified, please use path to local file'
                                    'or url of an image hosted on internet')

    dec_im, enc_im = encrypt_then_decrypt(locator, scheme)
    dec_im.show()
    enc_im.show()

    plot_channels(dec_im, enc_im, scheme)
