from multiprocessing import Process, Queue
from random import seed, random

import numpy as np
from PIL import Image

from util import pwlcm


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
