# Uzdevums: 1. Realizēt filtrēšanu ar Furjē transformācijas palīdzību
# Autors: Reinis Ričards Ramis
# Stud. apl. nr.: rr26072

import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def dft_1d(fr_in, fi_in):
    n = len(fr_in)
    Fr = [0.0] * n
    Fi = [0.0] * n
    for u in range(n):
        Fr[u] = Fi[u] = 0
        for k in range(n):
            angle = 2 * math.pi * u * k / n
            s = (-1) ** k
            Fr[u] += s * (fr_in[k] * math.cos(angle) + fi_in[k] * math.sin(angle))
            Fi[u] += s * (fi_in[k] * math.cos(angle) - fr_in[k] * math.sin(angle))
        Fr[u] /= n
        Fi[u] /= n
    return Fr, Fi

def idft_1d(Fr, Fi):
    n = len(Fr)
    fr = [0.0] * n
    fi = [0.0] * n
    for k in range(n):
        fr[k] = fi[k] = 0
        for u in range(n):
            angle = 2 * math.pi * u * k / n
            fr[k] += Fr[u] * math.cos(angle) - Fi[u] * math.sin(angle)
            fi[k] += Fr[u] * math.sin(angle) + Fi[u] * math.cos(angle)
        fr[k] *= (-1) ** k
        fi[k] *= (-1) ** k
    return fr, fi

def dft_2d(img):
    h, w = img.shape
    Rr = np.zeros((h, w))
    Ri = np.zeros((h, w))
    for y in range(h):
        Rr[y, :], Ri[y, :] = dft_1d(img[y, :], [0.0] * w)

    Fr = np.zeros((h, w))
    Fi = np.zeros((h, w))
    for x in range(w):
        Fr[:, x], Fi[:, x] = dft_1d(Rr[:, x], Ri[:, x])
    return Fr, Fi

def idft_2d(Fr, Fi):
    h, w = Fr.shape
    Rr = np.zeros((h, w))
    Ri = np.zeros((h, w))
    for y in range(h):
        Rr[y, :], Ri[y, :] = idft_1d(Fr[y, :], Fi[y, :])

    fr = np.zeros((h, w))
    for x in range(w):
        fr[:, x], _ = idft_1d(Rr[:, x], Ri[:, x])
    return fr

def low_pass(Fr, Fi, radius):
    h, w = Fr.shape
    y, x = np.indices((h, w))
    mask = np.sqrt((y - h // 2) ** 2 + (x - w // 2) ** 2) <= radius
    return Fr * mask, Fi * mask

# 1. Nolasa bildi
img = Image.open('sample.png').convert('L')
data = np.asarray(img, dtype=float)

# 2. Furjē transformācija
Fr, Fi = dft_2d(data)

# 3. Manipulācija ar transformācijas rezultātu
Fr, Fi = low_pass(Fr, Fi, radius=30)

# 4. Inversā Furjē transformācija
result = idft_2d(Fr, Fi)

# 5. Parādi rezultātu uz ekrāna
result = np.clip(result, 0, 255).astype(np.uint8)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(data, cmap='gray', vmin=0, vmax=255)
axes[0].set_title('Oriģināls')
axes[0].axis('off')

axes[1].imshow(result, cmap='gray', vmin=0, vmax=255)
axes[1].set_title('Filtrēts')
axes[1].axis('off')

plt.show()
