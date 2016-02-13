import numpy as np
from PIL import Image


# pixel-by-pixel by matrix mult
def first_method(name):
    img = Image.open(name)
    wRGB = np.array([0.299, 0.587, 0.114])[:, np.newaxis]
    wRGBA = np.array([0.299, 0.587, 0.114, 0])[:, np.newaxis]
    if np.array(img.getpixel((0, 0))).shape[0] == 4:
        weight = wRGBA
    else:
        weight = wRGB
    x, y = img.size
    for i in range(x):
        for j in range(y):
            col = np.array(img.getpixel((i, j))).dot(weight)
            img.putpixel((i, j), tuple([col, col, col, 255]))
    img.save('1' + name)
    img.close()


# with Pillow func
def second_method(name):
    img = Image.open(name)
    img = img.convert(mode="L")
    img.save('2' + name)
    img.close()


# pixel-by-pixel with ariphmetical method
def third_method(name):
    img = Image.open(name)
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            pixel = img.getpixel((i, j))
            pixel = int(0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[0])
            img.putpixel((i, j), tuple((pixel, pixel, pixel, 255)))
    img.save('3' + name)
    img.close()
