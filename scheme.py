#!/home/gregs0n/venvs/numpy-venv/bin/python3

import numpy as np
from scipy.sparse.linalg import *
from dataclasses import dataclass

from draw import drawHeatmap
from enviroment import Test, Material
from boundary import GetBoundary, stef_bolc

def debug_cg(xk:np.ndarray):
    n = int(np.sqrt(xk.size))
    data = xk.reshape((n, n))
    drawHeatmap(data, [0, 1], "x_k")

## FDM - First Discrete Method

@dataclass
class FDM:
    F: np.ndarray
    G: np.ndarray
    n: int
    h: np.float64
    limits: list[np.float64, np.float64]

    def __call__(self, H_:np.ndarray):
        H = H_.reshape((self.n, self.n))
        res = np.zeros_like(H)
        res[1:-1, 1:-1] = -1/self.h*(H[2:, 1:-1] + H[:-2, 1:-1] + H[1:-1, 2:] + H[1:-1, :-2] - 4*H[1:-1, 1:-1])
        
        res[ 1:-1,  0  ] = H[1:-1, 0] - H[1:-1, 1] - (H[2:, 0] - 2*H[1:-1, 0] + H[:-2, 0]) + H[1:-1, 0]
        res[  -1 , 1:-1] = H[-1, 1:-1] - H[-2, 1:-1] - (H[-1, 2:] - 2*H[-1, 1:-1] + H[-1, :-2]) + H[-1, 1:-1]
        res[ 1:-1, -1  ] = H[1:-1, -1] - H[1:-1, -2] - (H[2:, -1] - 2*H[1:-1, -1] + H[:-2, -1]) + H[1:-1, -1]
        res[   0 , 1:-1] = H[0, 1:-1] - H[1, 1:-1] - (H[0, 2:] - 2*H[0, 1:-1] + H[0, :-2]) + H[0, 1:-1]

        res[ 0,  0] = H[0, 0] - 0.5 * (H[1, 0] + H[0, 1]) + H[0, 0]
        res[-1,  0] = H[-1, 0] - 0.5 * (H[-2, 0] + H[-1, 1]) + H[-1, 0]
        res[-1, -1] = H[-1, -1] - 0.5 * (H[-2, -1] + H[-1, -2]) + H[-1, -1]
        res[ 0, -1] = H[0, -1] - 0.5 * (H[1, -1] + H[0, -2]) + H[0, -1]

        return res.reshape((self.n**2))
    
    def _solve(self):
        A = LinearOperator((self.n**2, self.n**2), matvec=self)
        res, exit_code = bicgstab(A,
              (self.F + self.G).reshape((self.n**2)),
              tol = 0.0, atol = 1.e-12,
              x0 = np.zeros(self.n**2) + 200.0
              )
        #print(exit_code)
        self.H = res
    
    def Solve_fdm(self):
        res = np.power(self.H/stef_bolc, 0.25)*100.0
        return res.reshape((self.n, self.n))
    
    def Solve_sdm(self, thermal_cond):
        res = self.SecondDiscrete(self.H, thermal_cond)
        return res.reshape((self.n, self.n))
    
    def SecondDiscrete(self, H_, thermal_cond):
        u_0 = np.power(H_/stef_bolc, 0.25)*100.0

        eps = self.h
        stef_bolc_loc = stef_bolc * 1e-8
        tcc = thermal_cond

        tc_e =  tcc/eps
        sq3 = np.sqrt(3.0)
        a3 = tcc/(4.0*eps*stef_bolc_loc)
        a = np.float_power(a3, 1.0/3.0)

        h_0 = np.pi * a / (6*sq3)

        H = lambda v: tc_e * (
            v + a*np.log(v**2 - a*v + a**2)/6.0 -
            a*np.arctan((2*v - a)/(sq3*a))/sq3 - 
            a*np.log(v + a)/3.0 - h_0
        )
        
        dH = lambda v: tc_e * (1.0 - a3/(a3 + v**3))
        
        u = u_0 - (H(u_0) - H_)/dH(u_0)
        err = np.abs(u - u_0).max()
        while err >= 1e-11:
            u_0 = u
            u = u_0 - (H(u_0) - H_)/dH(u_0)
            err = np.abs(u - u_0).max()
            print(err)
        return u


if __name__ == "__main__":
    cells = 20
    material = Material("template", .05)
    test = Test(0, [0, 0], material, cells)
    F, G = GetBoundary(test)
    #drawHeatmap(F, [0, 1], "F")
    #rawHeatmap(G, [0, 1], "G")
    #drawHeatmap(F+G, [0, 1], "F+G")
    fdm = FDM(F, G, cells, 1.0/cells, [0.0, 1.0])
    fdm._solve()
    res = fdm.Solve_sdm(material.thermal_cond)
    drawHeatmap(res, [0, 1], "computed")