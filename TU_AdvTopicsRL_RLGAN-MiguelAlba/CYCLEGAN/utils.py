import gzip
import pickle
import os
import glob
import imageio
import dill

from PIL import Image
import numpy as np


def slurpjson(fn):
    import json
    with open(fn, 'r') as f:
        return json.load(f)


def mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)
    return path


def mdir(path):
    try:
        os.makedirs(path)
        # print("Directory ", path, " Created ")
    except FileExistsError:
        print("Directory ", path, " already exists")


def save(fn, a):
    with gzip.open(fn, 'wb', compresslevel=2) as f:
        pickle.dump(a, f, 2)


def imread(fn):
    img = imageio.imread(fn)
    return img


def writeimg(fn, img, aspect_ratio=1.0):
    h, w, _ = img.shape
    img = Image.fromarray(img)
    # if aspect_ratio > 1:
    #     img = img.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    # elif aspect_ratio < 1:
    #     img = img.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    # else:
    img.save(fn)


def load(fn):
    with gzip.open(fn, 'rb') as f:
        return pickle.load(f)


def spit_json(fn, data):
    import json
    with open(fn, 'w') as outfile:
        json.dump(data, outfile)


def get_all_folder(path):
    return glob.glob('{}/*'.format(path))


def run_dill_encoded(payload):
    fun, args = dill.loads(payload)
    return fun(*args)


def display_image(image):
    import matplotlib.pyplot as plt
    from CYCLEGAN.GANs.net_utils import inverse_transform
    image = inverse_transform(image.numpy()[0])
    image = np.clip(image, 0, 255)
    plt.imshow(image)
    plt.show()
