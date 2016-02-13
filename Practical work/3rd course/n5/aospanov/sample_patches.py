import numpy as np


def normalize_data(images):
    mju = np.mean(images, axis=0)
    stnd = np.std(images, axis=0)
    left = mju - 3 * stnd
    right = mju + 3 * stnd
    tmp = np.minimum(np.maximum(images, left), right)
    tmp -= np.min(tmp, axis=0)
    tmp /= np.max(tmp, axis=0)
    tmp *= 0.8
    tmp += 0.1
    return tmp


def sample_patches_raw(images, num_patches=10000, patch_size=8):
    d = np.int(np.sqrt(images.shape[1] / 3))
    if d < patch_size:
        print('Error: patch_size is too large')
        return 0
    imgs = images.reshape(images.shape[0], d, d, 3)  # reshaped img

    # indxs [numOfPic, i, j, ...]
    indx = np.random.randint(0, images.shape[0], 3 * num_patches)
    indx[1::3] %= d - patch_size
    indx[2::3] %= d - patch_size

    # reshaped indxs [[numOfPic, i, j], [...], ...]
    indr = indx.reshape(indx.size // 3, 3)

    img = np.zeros((num_patches, 3 * patch_size * patch_size), dtype=np.uint8)
    for i in range(indr.shape[0]):
        img[i] = imgs[indr[i, 0], indr[i, 1]: indr[i, 1] + patch_size,
                      indr[i, 2]: indr[i, 2] + patch_size].flatten()  # reshape(3 * patch_size * patch_size)
    return img


def sample_patches(images, num_patches=10000, patch_size=8):
    return normalize_data(sample_patches_raw(images, num_patches, patch_size))
