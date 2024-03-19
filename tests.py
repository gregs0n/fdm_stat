#!/home/gregs0n/venvs/numpy-venv/bin/python3

from enviroment import *
from draw import *
from fdm_scheme import *
from sdm_scheme import *

import numpy as np


def main():
    start = 10
    finish = 45
    tccs = [1, 5, 10, 20, 50, 100, 200]
    errs = [[] for _ in tccs]
    for cells in range(start, finish + 1):
        material = Material("template", 5.12)
        test = Test(0, [0, 0], material, cells)
        sol_exact = [np.load(f"two_sin/{tcc}/{cells}.npy") for tcc in tccs]
        F, G = GetBoundary(test)
        #fdm = FDM(F, G, cells, 1.0 / cells, [0.0, 1.0], material)
        #sol_fdm = fdm.solve()
        for i in range(len(tccs)):
            sdm = SDM(
                F, G, cells, 1.0 / cells, [0.0, 1.0], Material("template", tccs[i])
            )
            sol_sdm = sdm.solve(1.0e-3, 300.0 * np.ones_like(F))
            err_loc = np.abs(sol_exact[i] - sol_sdm).max()
            errs[i].append(err_loc.max() / (sol_exact[i].max() - sol_exact[i].min()))
        print(f"{cells} : ready |")

    draw1D(
        list(map(np.array, errs)),
        [start, finish],
        "errors",
        legends=list(map(lambda s: f"$\lambda$={s}", tccs)),
        yscale="log",
        show_plot=1,
    )


if __name__ == "__main__":
    main()
