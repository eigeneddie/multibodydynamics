# import necessary packages
import numpy as np  
import matplotlib.pyplot as plt
import functionScript3 as fc # Important custom functions
from functionScript3 import ATransformMatrix as ATrans, QdCalc1Joint, QdCalc2Joint, link2index

# 1. === INPUT PARAMETERS ======
link1, link2, link3  = 1, 1, 1 # [m]

timeStart, timeEnd, stepTime   = 0, 10, 0.001  # [s] time start, end, increment

mass1, mass2, mass3 = 1.2, 1.2, 1.2 #[kg]

# initial conditions
theta2Init = np.pi/6 # [rad]
theta3Init = np.pi/3 # [rad]

gravity = 9.81 # [m/s^2]

u_bar_1A  = np.array( [[0],        [0]] )  
u_bar_2A  = np.array( [[0], [ link2/2]] )
u_bar_2B  = np.array( [[0], [-link2/2]] )
u_bar_3B  = np.array( [[0], [ link3/2]] )
u_bar_3C  = np.array( [[0], [-link3/2]] )


# 3. DEFINE GENERALIZED COORDINATES
qi       = np.zeros((3,1), dtype = float)# position
qiDot    = np.zeros((3,1), dtype = float)# velocity
qiDotDot = np.zeros((3,1), dtype = float)# acceleration



# dependent coordinates
q_dep         = np.zeros((4,1)) # position
qDot_dep      = np.zeros((4,1)) # velocity
qDotDot_dep   = np.zeros((4,1)) # acceleration

# independent coordinates
q_indep       = np.array( [[theta2Init], [theta3Init]] ) # position
qDot_indep    = np.zeros((2,1)) # velocity
qDotDot_indep = np.zeros((2,1)) # acceleration