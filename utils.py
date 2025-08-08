import fabio
import numpy as np
from scipy.ndimage import map_coordinates

DETECTOR_GAP = {
    "Eiger1M": [1065, 1030, [514, 550], []],
    "Eiger4M": [2167, 2070, [514, 550, 1065, 1101, 1616, 1652], [1030, 1039]],
    "Eiger9M": [
        3269, 3110, [514, 550, 1065, 1101, 1616, 1652, 2167, 2203, 2718, 2754],
        [1030, 1039, 2070, 2079]
    ],
    "Eiger16M": [
        4371, 4150,
        [
            514, 550, 1065, 1101, 1616, 1652, 2167, 2203, 2718, 2754, 3269,
            3305, 3820, 3856
        ], [1030, 1039, 2070, 2079, 3109, 3119]
    ],
    "Pilatus1M":
    [1043, 981, [195, 211, 407, 423, 619, 635, 831, 847], [487, 493]],
    "Pilatus2M": [
        1679, 1475,
        [
            195, 211, 407, 423, 619, 635, 831, 847, 1043, 1059, 1255, 1271,
            1467, 1483
        ], [487, 493, 981, 987]
    ],
    "Pilatus300K": [619, 487, [195, 211, 407, 423], []],
    "Pilatus300K-W": [195, 1475, [], [487, 493, 981, 987]],
}

DETECTOR_PIXEL = {
    "Eiger1M": 0.075,
    "Eiger4M": 0.075,
    "Eiger9M": 0.075,
    "Eiger16M": 0.075,
    "Pilatus1M": 0.172,
    "Pilatus2M": 0.172,
    "Pilatus300K": 0.172,
    "Pilatus300K-W": 0.172
}


def generate_detector_mask(detector_type):
    ny, nx, row, col = DETECTOR_GAP[detector_type]
    mask = np.zeros((ny, nx), dtype=np.int8)
    for i in range(0, len(row), 2):
        start_row = row[i]
        end_row = row[i + 1] + 1
        mask[start_row:end_row, :] = 1
    for j in range(0, len(col), 2):
        start_col = col[j]
        end_col = col[j + 1] + 1
        mask[:, start_col:end_col] = 1
    return mask


def get_detector_pixel_size(detector_type):
    return DETECTOR_PIXEL[detector_type]


def load_data(file_path):
    return fabio.open(file_path).data


def fill_gap(flatfield, mask, data0, data1, data2, x1, y1, x2, y2):
    data = data0 * flatfield
    gapmask = mask > 0
    data[gapmask] = 0
    first_fill = data1 * flatfield
    first_fill[gapmask] = 0
    yy, xx = np.where(gapmask)
    py = yy + y1
    px = xx + x1
    data[gapmask] = map_coordinates(first_fill, np.asarray([py, px]), order=1)
    if data2 is not None:
        mask = gapmask.copy().astype(np.float64)
        mask1 = mask.copy()
        mask1[gapmask] = map_coordinates(mask, np.asarray([py, px]), order=1)
        mask2 = mask1 > 0
        yy2, xx2 = np.where(mask2)
        py2 = yy2 + y2
        px2 = xx2 + x2
        data[mask2] = map_coordinates(data2 * flatfield,
                                      np.asarray([py2, px2]),
                                      order=1)
    return data
