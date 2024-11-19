#!/home/gregs0n/venvs/numpy-venv/bin/python3

import numpy as np
from scipy.sparse.linalg import *
from dataclasses import dataclass

from draw import drawHeatmap
from enviroment import Test, Material
from boundary import GetBoundary, stef_bolc, w


## FDM - First Discrete Method
@dataclass
class FDM:
    F: np.ndarray
    G: np.ndarray
    n: int
    h: np.float64
    limits: list[np.float64, np.float64]
    material: Material

    def operator(self, H_: np.ndarray):
        H = H_.reshape((self.n, self.n))
        res = np.zeros_like(H)

        #internal cells
        res[1:-1, 1:-1] = (
            -1
            / self.h
            * (
                H[2:, 1:-1]
                + H[:-2, 1:-1]
                + H[1:-1, 2:]
                + H[1:-1, :-2]
                - 4.0 * H[1:-1, 1:-1]
            )
        )

        #edge cells
        res[1:-1, 0] = (
            H[1:-1, 0]
            - H[1:-1, 1]
            - (H[2:, 0] - 2.0 * H[1:-1, 0] + H[:-2, 0])
            + H[1:-1, 0]
        )
        res[-1, 1:-1] = (
            H[-1, 1:-1]
            - H[-2, 1:-1]
            - (H[-1, 2:] - 2.0 * H[-1, 1:-1] + H[-1, :-2])
            + H[-1, 1:-1]
        )
        res[1:-1, -1] = (
            H[1:-1, -1]
            - H[1:-1, -2]
            - (H[2:, -1] - 2.0 * H[1:-1, -1] + H[:-2, -1])
            + H[1:-1, -1]
        )
        res[0, 1:-1] = (
            H[0, 1:-1]
            - H[1, 1:-1]
            - (H[0, 2:] - 2.0 * H[0, 1:-1] + H[0, :-2])
            + H[0, 1:-1]
        )

        #corner cells
        res[0, 0] = H[0, 0] - 0.5 * (H[1, 0] + H[0, 1]) + H[0, 0]
        res[-1, 0] = H[-1, 0] - 0.5 * (H[-2, 0] + H[-1, 1]) + H[-1, 0]
        res[-1, -1] = H[-1, -1] - 0.5 * (H[-2, -1] + H[-1, -2]) + H[-1, -1]
        res[0, -1] = H[0, -1] - 0.5 * (H[1, -1] + H[0, -2]) + H[0, -1]

        return res.reshape((self.n**2))

    def solve(self):
        A = LinearOperator((self.n**2, self.n**2), matvec=self.operator)
        res, exit_code = bicgstab(
            A,
            (self.F + self.G).reshape((self.n**2)),
            rtol=0.0,
            atol=1.0e-12,
            x0=np.zeros(self.n**2) + 200.0,
            # callback=self._bicg_callback
        )
        if exit_code:
            print(f"operator failed with exit code: {exit_code}")
            exit()
        U = (w * np.power(res / stef_bolc, 0.25)).reshape((self.n, self.n))
        return U
    
    def _bicg_callback(self, x_k: np.ndarray):
        tmp = (w * np.power(x_k / stef_bolc, 0.25)).reshape((self.n, self.n))
        nrm = self.h**2 * (
            np.sum(tmp[1:-1, 1:-1])
            + 0.5 * (
                np.sum(tmp[0, 1:-1])
                + np.sum(tmp[-1, 1:-1])
                + np.sum(tmp[1:-1, 0])
                + np.sum(tmp[1:-1, -1])
            )
            + 0.25 * (
                tmp[0, 0]
                + tmp[0, -1]
                + tmp[-1, 0]
                + tmp[-1, -1]
            )
        )

        print(nrm)


if __name__ == "__main__":
    cells = 20
    material = Material("template", 5.0)
    test = Test(0, [0, 0], material, cells)
    F, G = GetBoundary(test)
    # drawHeatmap(F, [0, 1], "F")
    #print(G[0, ::].max(), G[0, ::].min())
    #drawHeatmap(G, [0, 1], "G")
    #exit()
    fdm = FDM(F, G, cells, 1.0 / cells, [0.0, 1.0], material)
    res = fdm.solve()
    np.save("fdm_20sqs_sol", res)
    np.save("fdm_20sqs_G", G)
    drawHeatmap(res, [0, 1], "computed")
