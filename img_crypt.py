import io

from pathlib import Path
from PIL import Image


def make_bitmap(path, bw=False):
    path = Path(path)
    bmap_path = path.stem + '_bmap.bmp'
    im = Image.open(path)

    if bw:
        im = im.convert('L')

    im.save(bmap_path)


def convert_to_bin_array(im):
    def to_bit(data, i):
        base = int(i/8)
        shift = i % 8
        return (data[base] & (1 << shift)) >> shift

    data_bytes = io.BytesIO()
    im.save(data_bytes, format='BMP')
    data_bytes = data_bytes.getvalue()
    return [to_bit(data_bytes, i) for i in range(len(data_bytes) * 8)]


def convert_from_bin_array(bin_array):
    data_bytes = []
    size = len(bin_array)
    bin_array = [str(i) for i in bin_array]
    i = 0

    while i < size:
        data_bytes.append((int("0b" + "".join(bin_array[i:i + 8]), 2)))
        i += 8

    data_bytes = bytes(data_bytes)

    return Image.frombytes(io.BytesIO(data_bytes))


if __name__ == '__main__':
    #make_bitmap('img/lucjan.JPG', bw=True)
    im = Image.open('lucjan_bmap.bmp')
    lucjan_bin = convert_to_bin_array(im)

    new_im = convert_from_bin_array(lucjan_bin)
    new_im.show()
