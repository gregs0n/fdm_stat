
from enviroment import *
from draw import *
from scheme import *

import numpy as np
from scipy.integrate import quad

def main():
    start = 10
    finish = 45
    tccs = [1, 5, 10, 20, 50, 100, 200]
    errs = [[] for _ in tccs]
    for cells in range(start, finish+1):
        material = Material("template", 5.12)
        test = Test(0, [0, 0], material, cells)
        sol_exact = [np.load(f"two_sin/{tcc}/{cells}.npy") for tcc in tccs]
        #for i in range(len(sol_exact)):
        #    drawHeatmap(sol_exact[i], [0, 1], f"two_sin/{cells}_{tccs[i]}", zlim=(300, 600), show_plot=0)
        #    print(f"{cells}:{tccs[i]} ready")
        #continue
        F, G = GetBoundary(test)
        fdm = FDM(F, G, cells, 1.0/cells, [0.0, 1.0])
        fdm._solve()
        sol_fdm = fdm.Solve_fdm()
        for i in range(len(tccs)):
            #sol_sdm = fdm.Solve_sdm(tccs[i])
            err_loc = np.abs(sol_exact[i] - sol_fdm).max()
            errs[i].append(err_loc.max()/(sol_exact[i].max() - sol_exact[i].min()))
        print(cells, ": ready")
    
    draw1D(
        list(map(np.array, errs)),
        [start, finish],
        "errors",
        legends=list(map(lambda s: f"$\lambda$={s}", tccs)),
        yscale="log",
        show_plot=0)

if __name__ == "__main__":
    main()
