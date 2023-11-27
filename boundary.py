import numpy as np
from scipy.integrate import nquad, quad

from enviroment import Test, Material

stef_bolc = 5.67036713 * 1e8

def GetFunc(f_num, g_num):
    pass

def GetBoundary(test: Test) -> list:
    bnd, material, cells, limits = list(test._asdict().values())[1:]

    f, g = GetFunc(*bnd)

    h: np.float64 = limits[0] / cells
    F: np.ndarray = np.zeros((cells, cells))
    G: np.ndarray = np.zeros((cells, cells))

    for i in range(cells):
        x1 = i * h
        for j in range(cells):
            x2 = j * h
            F[i, j], _ = nquad(f, [[x1, x1+h], [x2, x2+h]])

    for k in range(1, cells - 1):
        x = k * h
        G[ 0,  k], _ = quad(g[0], x, x + h)
        G[ k, -1], _ = quad(g[1], x, x + h)
        G[-1,  k], _ = quad(g[2], x, x + h)
        G[ k,  0], _ = quad(g[3], x, x + h)

    G[ 0,  0] = 0.5 * (quad(g[0], 0, h)[0] + quad(g[3], 0, h)[0])
    G[ 0, -1] = 0.5 * (quad(g[0], limits[0]-h, limits[0])[0] + quad(g[1], 0, h)[0])
    G[-1, -1] = 0.5 * (quad(g[1], limits[1]-h, limits[1])[0] + quad(g[2], limits[0]-h, limits[0])[0])
    G[-1,  0] = 0.5 * (quad(g[2], 0, h)[0] + quad(g[3], limits[1]-h, limits[1])[0])

    return [F, G]

