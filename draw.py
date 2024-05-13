import numpy as np
import matplotlib.pyplot as plt


def draw1D(
    data: list, limits: list, plot_name: str, yscale="linear", show_plot=True, ylim=[], legends=[]
):

    arg = np.linspace(limits[0], limits[1], data[0].size)
    fig, ax = plt.subplots()
    ax.set_title(plot_name)
    colors = ["blue", "red", "green"]
    for i in range(len(data)):
        lab = legends[i] if legends else f"plot{i+1}"
        ax.plot(arg, data[i], label=lab)
    if not ylim:
        ylim = [min([i.min() for i in data]), max([i.max() for i in data])]
    ax.set_yscale(yscale)
    ax.set_ylim(ymin=ylim[0], ymax=ylim[1])
    ax.set_xlim(xmin=1.0/limits[0], xmax=1.0/limits[1])
    ax.grid(True)
    # ax.legend()
    if show_plot:
        plt.show()
    else:
        fig.savefig(plot_name + ".png", dpi=500)
    plt.close()
    del fig, ax


def draw2D(data: np.ndarray, limits: list, plot_name: str, show_plot=True, zlim=[]):
    n = data.shape[0]

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_title(plot_name)

    x = np.linspace(limits[0], limits[1], n)
    y = np.linspace(limits[0], limits[1], n)
    xgrid, ygrid = np.meshgrid(x, y)
    surf = ax.plot_surface(xgrid, ygrid, data, cmap="plasma")
    if not zlim:
        zlim = [data.min(), data.max()]
    # ax.zaxis.set_major_formatter('{x:.02f}')
    ax.set_zlim(zmin=zlim[0], zmax=zlim[1])
    fig.colorbar(surf, shrink=0.5, aspect=5)
    if show_plot:
        plt.show()
    else:
        fig.savefig(plot_name + ".png", dpi=500)
    plt.close()
    del fig, ax


def drawHeatmap(
    data: np.ndarray, limits: list, plot_name: str, show_plot=True, zlim=[]
):
    n = data.shape[0]
    fig, ax = plt.subplots()
    h = (limits[1] - limits[0])/n
    x = np.arange(limits[0] + h/2, limits[1], h)
    y = np.arange(limits[0] + h/2, limits[1], h)
    ## x, h = np.linspace(limits[0], limits[1], n, retstep=True)
    ## y = np.linspace(limits[0], limits[1], n)
    xgrid, ygrid = np.meshgrid(x, y)
    if not zlim:
        zlim = [data.min(), data.max()]
    c = ax.pcolormesh(
        xgrid, ygrid, data, shading="nearest", cmap="RdBu_r", vmin=zlim[0], vmax=zlim[1]
    )  # cmap='hot' | 'afmhot' | 'gist_heat'
    ax.set_title(plot_name)
    ax.axis([limits[0], limits[1], limits[0], limits[1]])
    #ax.plot(xgrid.flat, ygrid.flat, '.', color='black')
    fig.colorbar(c, ax=ax)
    if show_plot:
        plt.show()
    else:
        fig.savefig(plot_name + ".png", dpi=1000)
    plt.close()
    del fig, ax

if __name__ == "__main__":
    f = lambda x, y: x*y
    n = 50
    h = 1.0 / n
    data = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            data[i, j] = f(i*h, j*h)
    drawHeatmap(data, [0, 1], "plot")