#!/home/gregs0n/venvs/numpy-venv/bin/python3

import numpy as np
from scipy.sparse.linalg import *

from draw import drawHeatmap
from enviroment import Test, Material
from boundary import GetBoundary, stef_bolc, w


def createH(eps, tcc):
    a = eps / tcc
    b = np.float_power(4.0 * stef_bolc / w * a, 1.0 / 3.0)
    b2 = b * b
    sq3 = np.sqrt(3.0)
    h_0 = np.pi / (6.0 * sq3 * b)

    H = (
        lambda v: w
        / a
        * (
            v
            + np.log(b2 * v**2 - b * v + 1.0) / (6.0 * b)
            - np.arctan((2.0 * b * v - 1.0) / sq3) / (sq3 * b)
            - np.log(b * v + 1.0) / (3.0 * b)
            - h_0
        )
    )

    dH = lambda v: 4.0 * stef_bolc / w * v**3 / (1.0 + 4.0 * stef_bolc / w * a * v**3)

    return H, dH


def createH2(eps, tcc):
    H = lambda v: stef_bolc * v**4

    dH = lambda v: 4.0 * stef_bolc / w * v**3

    return H, dH



## SDM - Second Discrete Method
class SDM:
    def __init__(
        self,
        F: np.ndarray,
        G: np.ndarray,
        n: int,
        h: np.float64,
        limits: list[np.float64, np.float64],
        material: Material,
    ):
        self.F = F
        self.G = G
        self.n = n
        self.h = h
        self.limits = limits
        self.material = Material

        self.H, self.dH = createH(self.h, material.thermal_cond)
        self.B, self.dB = createH(self.h, 2.0 * material.thermal_cond)

    def operator(self, u_: np.ndarray):
        u = u_.reshape((self.n, self.n))
        res = np.zeros_like(u)
        H, B = self.H, self.B

        # internal cells
        res[1:-1, 1:-1] = (
            -1
            / self.h
            * (
                H(u[2:, 1:-1])
                + H(u[:-2, 1:-1])
                + H(u[1:-1, 2:])
                + H(u[1:-1, :-2])
                - 4.0 * H(u[1:-1, 1:-1])
            )
        )

        # edge cells
        res[1:-1, 0] = (
            H(u[1:-1, 0])
            - H(u[1:-1, 1])
            - (H(u[2:, 0]) - 2.0 * H(u[1:-1, 0]) + H(u[:-2, 0]))
            + B(u[1:-1, 0])
        )
        res[-1, 1:-1] = (
            H(u[-1, 1:-1])
            - H(u[-2, 1:-1])
            - (H(u[-1, 2:]) - 2.0 * H(u[-1, 1:-1]) + H(u[-1, :-2]))
            + B(u[-1, 1:-1])
        )
        res[1:-1, -1] = (
            H(u[1:-1, -1])
            - H(u[1:-1, -2])
            - (H(u[2:, -1]) - 2.0 * H(u[1:-1, -1]) + H(u[:-2, -1]))
            + B(u[1:-1, -1])
        )
        res[0, 1:-1] = (
            H(u[0, 1:-1])
            - H(u[1, 1:-1])
            - (H(u[0, 2:]) - 2.0 * H(u[0, 1:-1]) + H(u[0, :-2]))
            + B(u[0, 1:-1])
        )

        # corner cells
        res[0, 0] = H(u[0, 0]) - 0.5 * (H(u[1, 0]) + H(u[0, 1])) + B(u[0, 0])
        res[-1, 0] = H(u[-1, 0]) - 0.5 * (H(u[-2, 0]) + H(u[-1, 1])) + B(u[-1, 0])
        res[-1, -1] = H(u[-1, -1]) - 0.5 * (H(u[-2, -1]) + H(u[-1, -2])) + B(u[-1, -1])
        res[0, -1] = H(u[0, -1]) - 0.5 * (H(u[1, -1]) + H(u[0, -2])) + B(u[0, -1])

        return res.reshape((self.n**2))

    def jacobian(self, du_: np.ndarray):
        u = self.U.reshape((self.n, self.n))
        du = du_.reshape((self.n, self.n))
        res = np.zeros_like(u)
        dH, dB = self.dH, self.dB

        # internal cells
        res[1:-1, 1:-1] = (
            -1
            / self.h
            * (
                dH(u[2:, 1:-1]) * w * du[2:, 1:-1]
                + dH(u[:-2, 1:-1]) * w * du[:-2, 1:-1]
                + dH(u[1:-1, 2:]) * w * du[1:-1, 2:]
                + dH(u[1:-1, :-2]) * w * du[1:-1, :-2]
                - 4.0 * dH(u[1:-1, 1:-1]) * w * du[1:-1, 1:-1]
            )
        )

        # edge cells
        res[1:-1, 0] = (
            dH(u[1:-1, 0]) * w * du[1:-1, 0]
            - dH(u[1:-1, 1]) * w * du[1:-1, 1]
            - (
                dH(u[2:, 0]) * w * du[2:, 0]
                - 2.0 * dH(u[1:-1, 0]) * w * du[1:-1, 0]
                + dH(u[:-2, 0]) * w * du[:-2, 0]
            )
            + dB(u[1:-1, 0]) * w * du[1:-1, 0]
        )
        res[-1, 1:-1] = (
            dH(u[-1, 1:-1]) * w * du[-1, 1:-1]
            - dH(u[-2, 1:-1]) * w * du[-2, 1:-1]
            - (
                dH(u[-1, 2:]) * w * du[-1, 2:]
                - 2.0 * dH(u[-1, 1:-1]) * w * du[-1, 1:-1]
                + dH(u[-1, :-2]) * w * du[-1, :-2]
            )
            + dB(u[-1, 1:-1]) * w * du[-1, 1:-1]
        )
        res[1:-1, -1] = (
            dH(u[1:-1, -1]) * w * du[1:-1, -1]
            - dH(u[1:-1, -2]) * w * du[1:-1, -2]
            - (
                dH(u[2:, -1]) * w * du[2:, -1]
                - 2.0 * dH(u[1:-1, -1]) * w * du[1:-1, -1]
                + dH(u[:-2, -1]) * w * du[:-2, -1]
            )
            + dB(u[1:-1, -1]) * w * du[1:-1, -1]
        )
        res[0, 1:-1] = (
            dH(u[0, 1:-1]) * w * du[0, 1:-1]
            - dH(u[1, 1:-1]) * w * du[1, 1:-1]
            - (
                dH(u[0, 2:]) * w * du[0, 2:]
                - 2.0 * dH(u[0, 1:-1]) * w * du[0, 1:-1]
                + dH(u[0, :-2]) * w * du[0, :-2]
            )
            + dB(u[0, 1:-1]) * w * du[0, 1:-1]
        )

        # corner cells
        res[0, 0] = (
            dH(u[0, 0]) * w * du[0, 0]
            - 0.5 * (dH(u[1, 0]) * w * du[1, 0] + dH(u[0, 1]) * w * du[0, 1])
            + dB(u[0, 0]) * w * du[0, 0]
        )
        res[-1, 0] = (
            dH(u[-1, 0]) * w * du[-1, 0]
            - 0.5 * (dH(u[-2, 0]) * w * du[-2, 0] + dH(u[-1, 1]) * w * du[-1, 1])
            + dB(u[-1, 0]) * w * du[-1, 0]
        )
        res[-1, -1] = (
            dH(u[-1, -1]) * w * du[-1, -1]
            - 0.5 * (dH(u[-2, -1]) * w * du[-2, -1] + dH(u[-1, -2]) * w * du[-1, -2])
            + dB(u[-1, -1]) * w * du[-1, -1]
        )
        res[0, -1] = (
            dH(u[0, -1]) * w * du[0, -1]
            - 0.5 * (dH(u[1, -1]) * w * du[1, -1] + dH(u[0, -2]) * w * du[0, -2])
            + dB(u[0, -1]) * w * du[0, -1]
        )

        return res.reshape((self.n**2))

    def solve(self, eps: np.float64, u_0: np.ndarray):
        self.U = u_0.reshape((self.n**2))
        A = LinearOperator((self.n**2, self.n**2), matvec=self.jacobian)
        b = (self.F + self.G).reshape((self.n**2))
        R = b - self.operator(self.U)
        dU, exit_code = bicgstab(
            A,
            R,
            rtol=1.0e-9,
            atol=1.0e-12,
            x0=R,
            # callback=self._bicg_callback
        )
        #drawHeatmap(self.U.reshape((self.n, self.n)), [0, 1], "initial guess")
        if exit_code:
            print(f"jacobian failed with exit code: {exit_code}")
            exit()
        err = np.abs(dU).max()
        print(f"{err:.3e}")
        while err > eps:
            #drawHeatmap(self.U.reshape((self.n, self.n)), [0, 1], "current solution")
            self.U += dU
            R = b - self.operator(self.U)
            dU, exit_code = bicgstab(
                A,
                R,
                rtol=1.0e-9,
                atol=1.0e-12,
                x0=dU,
                # callback=self._bicg_callback
            )
            if exit_code:
                print(f"jacobian failed with exit code: {exit_code}")
                exit()
            err = np.abs(dU).max()
            print(f"{err:.3e}")
        self.U *= w
        return self.U.reshape((self.n, self.n))
    
    def _bicg_callback(self, x_k: np.ndarray):
        tmp = w * x_k.reshape((self.n, self.n))
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

        print(f"\t{nrm:.3e}")


if __name__ == "__main__":
    cells = 20
    tcc = 1.0

    h = 1.0 / cells
    material = Material("template", tcc)
    H, dH = createH(h, 2*tcc)
    test = Test(0, [0, 0], material, cells)
    F, G = GetBoundary(test, H)
    # drawHeatmap(F, [0, 1], "F")
    # print(G[0, ::].max(), G[0, ::].min())
    # drawHeatmap(G, [0, 1], "G")
    # exit()
    sdm = SDM(F, G, cells, h, [0.0, 1.0], material)
    res = sdm.solve(1.0e-9, 300.0 / w * np.ones_like(F))
    np.save("sdm_20sqs_tcc01_sol", res)
    np.save("sdm_20sqs_tcc01_G", G)
    drawHeatmap(res, [0, 1], "computed")
