import numpy as np  
import matplotlib.pyplot as plt
import calcModule as fc # Important custom functions
import constraintModule as cmod
from doublePendulu import u_bar_2A, u_bar_2B, u_bar_3B, u_bar_3C, qi

def config():
    r2A = fc.local2global(qi, u_bar_2A, 2)
    r2B = fc.local2global(qi, u_bar_2B, 2)
    r3B = fc.local2global(qi, u_bar_3B, 3)
        
        # 4. CONSTRAINT EQUATION C
    constraintVect = cmod.constraintEquation(r2A, r2B, r3B) #, r3C, timeNow)

        # 5. JACOBIAN MATRIX Cq
    Cq, Cq_dep, Cq_indep = cmod.jacobianMatrix(qi, constraintVect, 
                                        u_bar_2A, u_bar_2B, u_bar_3B)

    return Cq, Cq_dep, Cq_indep