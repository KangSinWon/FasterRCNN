import numpy as np
from label_dataset import Label

def read_calib(filepath):
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line)==0: continue
            key, value = line.split(':', 1)
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data


def read_label(filepath):
    lines = [line.rstrip() for line in open(filepath)]
    objects = [Label(line) for line in lines]
    return objects


def read_velo(filepath):
    velo = np.fromfile(filepath, dtype=np.float32)
    velo = velo.reshape((-1, 4))
    return velo

def rotation_y(t):
    cosine = np.cos(t)
    sine = np.sin(t)

    return np.array([[cosine, 0, sine], 
                     [0, 1, 0], 
                     [-sine, 0, cosine]])
                    