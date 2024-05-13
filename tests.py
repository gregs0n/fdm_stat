#!/home/gregs0n/venvs/numpy-venv/bin/python3

from enviroment import *
from fdm_scheme import *
from sdm_scheme import *

import numpy as np
import matplotlib.pyplot as plt

start = 10
finish = 50
tccs = [1, 5, 10, 20]#, 50, 100, 200]
cells_range = list(range(start, finish+1))

def main():    
    errs = {tcc: [[], []] for tcc in tccs}
    for cells in cells_range:
        h = 1.0 / cells
        material = Material("template", 5.12)
        test = Test(0, [0, 0], material, cells)
        sol_exact = [np.load(f"two_sin/{tcc}/{cells}.npy") for tcc in tccs]
        F, G = GetBoundary(test)
        fdm = FDM(F, G, cells, h, [0.0, 1.0], material)
        sol_fdm = fdm.solve()
        for i in range(len(tccs)):
            print(f"\t---{tccs[i]:02d}---")
            H = createH(h, 2*tccs[i])[0]
            F, G = GetBoundary(test, H)
            sdm = SDM(
               F, G, cells, h, [0.0, 1.0], Material("template", tccs[i])
            )
            sol_sdm = sdm.solve(1.0e-9, 300.0 * np.ones_like(F))
            temp_length = sol_exact[i].max() - sol_exact[i].min()
            errs[tccs[i]][0].append(np.abs(sol_exact[i] - sol_fdm).max() / temp_length)
            errs[tccs[i]][1].append(np.abs(sol_exact[i] - sol_sdm).max() / temp_length)
        print(f"{cells} : ready |")

    data = []
    legends = []
    for i in range(len(tccs)):
        tcc = tccs[i]
        data += list(map(np.array, errs[tcc]))
        legends += [f"FDM:$\lambda$={tcc}", f"SDM:$\lambda$={tcc}"]

    draw1D(
        data,
        [start, finish],
        f"fdm & sdm errors",
        legends=legends,
        yscale="log",
        show_plot=1,
    )
    save_computations = input("Save computations? (y/n):") == "y"
    if save_computations:
        for i in range(len(tccs)):
            tcc = tccs[i]
            np.save(f"errs/two_sin/lambda_{tcc:03d}:FDM", data[2*i])
            np.save(f"errs/two_sin/lambda_{tcc:03d}:SDM", data[2*i+1])


def draw1D(
    data: list,
    limits: list,
    plot_name: str,
    yscale="linear",
    show_plot=True,
    ylim=[],
    legends=[],
):

    arg = np.linspace(limits[0], limits[1], data[0].size)
    fig, ax = plt.subplots()
    ax.set_title(plot_name)
    colors = ["blue", "orange", "green", "red", "purple", "brown", "pink"]

    for i in range(len(data)):
        lab = legends[i] if legends else f"plot_{i+1}"
        ax.plot(
            (1.0 / arg),
            data[i],
            label=lab,
            color=colors[i//2],
            #marker="o" if i%2 == 0 else "s",
            linestyle='-' if i%2 == 0 else '--',
            #linewidth=1.25,
        )
    if not ylim:
        ylim = [min([i.min() for i in data]), max([i.max() for i in data])]
    ax.set_yscale(yscale)
    ax.set_ylim(ymin=ylim[0], ymax=ylim[1])
    ax.set_xlim(xmin=1.0 / limits[0], xmax=1.0 / limits[1])
    ax.grid(True)
    ax.legend()
    if show_plot:
        plt.show()
    else:
        fig.savefig(plot_name + ".png", dpi=500)
    plt.close()
    del fig, ax

def draw_computed():
    data = []
    legends = []
    #file1 = open("fdm_err_2.txt", "w")
    #file2 = open("sdm_err_2.txt", "w")
    for i in range(len(tccs)):
        tcc = tccs[i]
        data += [
            np.load(f"errs/two_sin/lambda_{tcc:03d}:FDM.npy"),
            np.load(f"errs/two_sin/lambda_{tcc:03d}:SDM.npy"),
        ]
        legends += [f"FDM:$\lambda$={tcc}", f"SDM:$\lambda$={tcc}"]
        #print(tcc, file=file1)
        #print(tcc, file=file2)
        #print(*cells_range, sep='\t', file=file1)
        #print(*cells_range, sep='\t', file=file2)
        #print(*[f"{data[-2][j]:.3e}" for j in range(len(data[-2]))], sep='\t', file=file1)
        #print(*[f"{data[-1][j]:.3e}" for j in range(len(data[-1]))], sep='\t', file=file2)
    #file1.close()
    #file2.close()
    draw1D(
        data,
        [start, finish],
        f"fdm & sdm errors",
        legends=legends,
        yscale="log",
        show_plot=1,
    )

if __name__ == "__main__":
    #main()
    draw_computed()