import numpy as np
from scipy.integrate import nquad, quad

from enviroment import Test, Material
from draw import draw1D

to_norm = 1
w = 100 if to_norm else 1
stef_bolc = 5.67036713 if to_norm else 5.67036713e-8

HeatStream = lambda t: stef_bolc * np.power(t, 4)


def GetFunc(f_num, g_num):
    ## returns f(x, y) and list of 4 g(t) NORMED
    tmax = 600.0 / w  # 500K, actually
    tmin = 300.0 / w
    coef = tmax - tmin
    d = tmin
    f = lambda x, y: 0.0
    g = [
        lambda t: (
            (d + coef * np.sin(np.pi * t / 0.3))
            if 0.0 <= t <= 0.3
            else tmin
        ),
        #lambda t: tmax,
        lambda t: tmin,
        lambda t: (
            (d + coef * np.sin(np.pi * (1.0 - t) / 0.3))
            if 0.7 <= t <= 1.0
            else tmin
        ),
        lambda t: tmin,
        #lambda t: tmin,
    ]
    return f, g


def GetBoundary(test: Test, __Hs=HeatStream) -> list:
    bnd, material, cells, limits = list(test._asdict().values())[1:]
    
    f_raw, g_raw = GetFunc(*bnd)
    
    f = lambda x, y: __Hs(f_raw(x, y))
    __x = np.linspace(0, 1, 1001)
    __g = np.array([g_raw[0](t) for t in __x])
    draw1D([__g], [0, 1], "plot")
    print(__g.min(), __g.max())
    g = [
        lambda t: __Hs(g_raw[0](t)),
        lambda t: __Hs(g_raw[1](t)),
        lambda t: __Hs(g_raw[2](t)),
        lambda t: __Hs(g_raw[3](t)),
    ]
    __g2 = np.array([g[0](t) for t in __x])
    draw1D([__g2], [0, 1], "plot")
    print(__g2.min(), __g2.max())

    h: np.float64 = (limits[1] - limits[0]) / cells
    F: np.ndarray = np.zeros((cells, cells))
    G: np.ndarray = np.zeros((cells, cells))

    #for i in range(cells):
    #    x1 = i * h
    #    for j in range(cells):
    #        x2 = j * h
    #        F[i, j], _ = nquad(f, [[x1, x1 + h], [x2, x2 + h]])

    for k in range(1, cells - 1):
        x = k * h
        G[k, 0], _ = quad(g[0], x, x + h)
        G[-1, k], _ = quad(g[1], x, x + h)
        G[k, -1], _ = quad(g[2], x, x + h)
        G[0, k], _ = quad(g[3], x, x + h)

    G[0, 0] = 0.5 * (
        quad(g[0], limits[0], limits[0] + h)[0]
        + quad(g[3], limits[0], limits[0] + h)[0]
    )
    G[-1, 0] = 0.5 * (
        quad(g[0], limits[1] - h, limits[1])[0]
        + quad(g[1], limits[0], limits[0] + h)[0]
    )
    G[-1, -1] = 0.5 * (
        quad(g[1], limits[1] - h, limits[1])[0]
        + quad(g[2], limits[1] - h, limits[1])[0]
    )
    G[0, -1] = 0.5 * (
        quad(g[2], limits[0], limits[0] + h)[0]
        + quad(g[3], limits[1] - h, limits[1])[0]
    )

    F *= 1.0 / h**2
    G *= 1.0 / h

    return [F, G]
