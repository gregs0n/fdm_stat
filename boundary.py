import numpy as np
from scipy.integrate import nquad, quad

from enviroment import Test, Material

stef_bolc = 5.67036713

HeatStream = lambda t: stef_bolc * np.power(t, 4)

def GetFunc(f_num, g_num):
    ## returns f(x, y) and list of 4 g(t) NORMED
    tmax = 6.0 #500K, actually
    tmin = 3.0
    coef = tmax - tmin
    d = tmin
    f = lambda x, y: 0.0
    g = [
        lambda t: HeatStream(d + coef * np.sin(np.pi * t / 0.3)) if 0.0 <= t <= 0.3 else HeatStream(tmin),
        #lambda t: HeatStream(tmax),
        lambda t: HeatStream(tmin),
        lambda t: HeatStream(tmin),#HeatStream(d + coef * np.sin(np.pi * (1.0 - t) / 0.3)) if 0.7 <= t <= 1.0 else HeatStream(tmin),
        lambda t: HeatStream(tmin),
    ]
    return f, g

def GetBoundary(test: Test) -> list:
    bnd, material, cells, limits = list(test._asdict().values())[1:]

    f, g = GetFunc(*bnd)

    h: np.float64 = (limits[1] - limits[0]) / cells
    F: np.ndarray = np.zeros((cells, cells))
    G: np.ndarray = np.zeros((cells, cells))

    for i in range(cells):
        x1 = i * h
        for j in range(cells):
            x2 = j * h
            F[i, j], _ = nquad(f, [[x1, x1+h], [x2, x2+h]])

    for k in range(1, cells - 1):
        x = k * h
        G[ k,  0], _ = quad(g[0], x, x + h)
        G[-1,  k], _ = quad(g[1], x, x + h)
        G[ k, -1], _ = quad(g[2], x, x + h)
        G[ 0,  k], _ = quad(g[3], x, x + h)

    G[ 0,  0] = 0.5 * (quad(g[0], limits[0], limits[0]+h)[0] + quad(g[3], limits[0], limits[0]+h)[0])
    G[-1,  0] = 0.5 * (quad(g[0], limits[1]-h, limits[1])[0] + quad(g[1], limits[0], limits[0]+h)[0])
    G[-1, -1] = 0.5 * (quad(g[1], limits[1]-h, limits[1])[0] + quad(g[2], limits[1]-h, limits[1])[0])
    G[ 0, -1] = 0.5 * (quad(g[2], limits[0], limits[0]+h)[0] + quad(g[3], limits[1]-h, limits[1])[0])

    F *= 1.0/h**2
    G *= 1.0/h

    return [F, G]
