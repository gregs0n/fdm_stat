#!/home/gregs0n/venvs/numpy-venv/bin/python3

import numpy as np
from scipy.sparse.linalg import *

from draw import drawHeatmap
from enviroment import Test, Material
from boundary import GetBoundary, stef_bolc


def createH(eps, tcc):
    stef_bolc_loc = stef_bolc * 1.0e-8

    a = eps / tcc
    b = np.float_power(4.0 * stef_bolc_loc * a, 1.0 / 3.0)
    b2 = b * b
    sq3 = np.sqrt(3.0)
    h_0 = np.pi / (6.0 * sq3 * b)

    H = (
        lambda v: 1.0
        / a
        * (
            v
            + np.log(b2 * v**2 - b * v + 1.0) / (6.0 * b)
            - np.arctan((2.0 * b * v - 1) / sq3) / (sq3 * b)
            - np.log(b * v + 1) / (3.0 * b)
            - h_0
        )
    )

    dH = lambda v: 4.0 * stef_bolc_loc * v**3 / (1.0 + 4.0 * stef_bolc_loc * a * v**3)

    return H, dH


# def createH(eps, tcc):
#    stef_bolc_loc = stef_bolc * 1.0e-8
#
#    H = lambda v: stef_bolc_loc * v**4
#
#    dH = lambda v: 4.0 * stef_bolc_loc * v**3
#
#    return H, dH


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
                dH(u[2:, 1:-1]) * du[2:, 1:-1]
                + dH(u[:-2, 1:-1]) * du[:-2, 1:-1]
                + dH(u[1:-1, 2:]) * du[1:-1, 2:]
                + dH(u[1:-1, :-2]) * du[1:-1, :-2]
                - 4.0 * dH(u[1:-1, 1:-1]) * du[1:-1, 1:-1]
            )
        )

        # edge cells
        res[1:-1, 0] = (
            dH(u[1:-1, 0]) * du[1:-1, 0]
            - dH(u[1:-1, 1]) * du[1:-1, 1]
            - (
                dH(u[2:, 0]) * du[2:, 0]
                - 2.0 * dH(u[1:-1, 0]) * du[1:-1, 0]
                + dH(u[:-2, 0]) * du[:-2, 0]
            )
            + dB(u[1:-1, 0]) * du[1:-1, 0]
        )
        res[-1, 1:-1] = (
            dH(u[-1, 1:-1]) * du[-1, 1:-1]
            - dH(u[-2, 1:-1]) * du[-2, 1:-1]
            - (
                dH(u[-1, 2:]) * du[-1, 2:]
                - 2.0 * dH(u[-1, 1:-1]) * du[-1, 1:-1]
                + dH(u[-1, :-2]) * du[-1, :-2]
            )
            + dB(u[-1, 1:-1]) * du[-1, 1:-1]
        )
        res[1:-1, -1] = (
            dH(u[1:-1, -1]) * du[1:-1, -1]
            - dH(u[1:-1, -2]) * du[1:-1, -2]
            - (
                dH(u[2:, -1]) * du[2:, -1]
                - 2.0 * dH(u[1:-1, -1]) * du[1:-1, -1]
                + dH(u[:-2, -1]) * du[:-2, -1]
            )
            + dB(u[1:-1, -1]) * du[1:-1, -1]
        )
        res[0, 1:-1] = (
            dH(u[0, 1:-1]) * du[0, 1:-1]
            - dH(u[1, 1:-1]) * du[1, 1:-1]
            - (
                dH(u[0, 2:]) * du[0, 2:]
                - 2.0 * dH(u[0, 1:-1]) * du[0, 1:-1]
                + dH(u[0, :-2]) * du[0, :-2]
            )
            + dB(u[0, 1:-1]) * du[0, 1:-1]
        )

        # corner cells
        res[0, 0] = (
            dH(u[0, 0]) * du[0, 0]
            - 0.5 * (dH(u[1, 0]) * du[1, 0] + dH(u[0, 1]) * du[0, 1])
            + dB(u[0, 0]) * du[0, 0]
        )
        res[-1, 0] = (
            dH(u[-1, 0]) * du[-1, 0]
            - 0.5 * (dH(u[-2, 0]) * du[-2, 0] + dH(u[-1, 1]) * du[-1, 1])
            + dB(u[-1, 0]) * du[-1, 0]
        )
        res[-1, -1] = (
            dH(u[-1, -1]) * du[-1, -1]
            - 0.5 * (dH(u[-2, -1]) * du[-2, -1] + dH(u[-1, -2]) * du[-1, -2])
            + dB(u[-1, -1]) * du[-1, -1]
        )
        res[0, -1] = (
            dH(u[0, -1]) * du[0, -1]
            - 0.5 * (dH(u[1, -1]) * du[1, -1] + dH(u[0, -2]) * du[0, -2])
            + dB(u[0, -1]) * du[0, -1]
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
            rtol=1.0e-3,
            atol=1.0e-6,
            x0=R,
        )
        if exit_code:
            print(f"jacobian failed with exit code: {exit_code}")
            exit()
        err = np.abs(dU).max()
        while err > eps:
            self.U += dU
            R = b - self.operator(self.U)
            dU, exit_code = bicgstab(
                A,
                R,
                rtol=1.0e-3,
                atol=1.0e-6,
                x0=dU,
            )
            if exit_code:
                print(f"jacobian failed with exit code: {exit_code}")
                exit()
            err = np.abs(dU).max()
        return self.U.reshape((self.n, self.n))


if __name__ == "__main__":
    cells = 20
    material = Material("template", 50.0)
    test = Test(0, [0, 0], material, cells)
    F, G = GetBoundary(test)
    sdm = SDM(F, G, cells, 1.0 / cells, [0.0, 1.0], material)
    res = sdm.solve(1.0e-4, 300.0 * np.ones_like(F))
    drawHeatmap(res, [0, 1], "computed")
