from random import seed, random

import numpy as np
from PIL import Image

from util import pwlcm


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


def cml_encrypt_channel(channel, p, iterations, cycles):
    pixels_flat = channel.flat
    pixel_num = len(pixels_flat)
    rand = [random() for _ in range(cycles * pixel_num)]

    cml_algo_encrypt(pixels_flat, pixel_num, rand, p, cycles, iterations)


def cml_decrypt_channel(channel, p, iterations, cycles):
    pixels_flat = channel.flat
    pixel_num = len(pixels_flat)
    rand = [random() for _ in range(cycles * pixel_num)]

    cml_algo_decrypt(pixels_flat, pixel_num, rand, p, cycles, iterations)


def cml_algo_encrypt(pixels_flat, pixel_num, rand, p, cycles, iterations):
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


def cml_algo_decrypt(pixels_flat, pixel_num, rand, p, cycles, iterations):
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
