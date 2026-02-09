import numpy as np

def I_formation(n, spacing=5):
    return np.array([[i * spacing, 0] for i in range(n)])

def V_formation(n, spacing=5):
    mid = n // 2
    coords = []
    for i in range(n):
        dx = abs(i - mid) * spacing
        dy = (i - mid) * spacing
        coords.append([dx, dy])
    return np.array(coords)

def O_formation(n, radius=20):
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.stack([radius*np.cos(angles),
                     radius*np.sin(angles)], axis=1)

def get_target_positions(name, n):
    if name == "I":
        return I_formation(n)
    if name == "V":
        return V_formation(n)
    if name == "O":
        return O_formation(n)
