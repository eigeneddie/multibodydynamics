# doin some importing 
import numpy as np  
import matplotlib.pyplot  as plt

import calcModule3D       as calMod
import forceModule3D      as fMod
import constraintModule3D as cnMod

from calcModule3D import link2index as l2i


# 1. === USER INPUT PARAMETERS (GLOBAL VARIABLES) ======

# ======== Car body parameters ===============
mass_body, mass_axle1, mass_axle2  = 1000,250,250 # [kg]
timeStart, timeEnd, stepSize = 0,60,0.1 # [s] 
time = np.arange(timeStart, timeEnd, stepSize)
n=7 # generalized coordinate
epsilon = 0.000000000001
gravity = 9.81 # [m/s^2]

# car dimensions
wheel_diameter = 0.61 # [m]
axleLength     = 1.8  # [m]
axleDistance   = 2.8  #[m] distance betwen front and rear
bodyHeight     = 1.4  # [m]

# springs and dampers
k_FLW, k_FRW, k_RLW, k_RRW = 210000, 210000, 210000, 210000 # N/m
c_FLW, c_FRW, c_RLW, c_RRW = 2000, 2000, 2000, 2000

# Derived parameters
Ixx = 1/12*mass_body*(  axleLength**2 + bodyHeight**2)
Iyy = 1/12*mass_body*(axleDistance**2 + bodyHeight**2)
Izz = 1/12*mass_body*(axleDistance**2 + axleLength**2)
# mtheta2 = GiT*I*Gi
Itheta2 = np.array([[Ixx, 0, 0],
                   [0, Iyy, 0],
                   [0, 0, Izz]])
massRR = np.array([[mass_body, 0, 0],
                   [0, mass_body, 0],
                   [0, 0, mass_body]])


# initial conditions
Ry1 = 0.3 # [m]

# Generalized coordinates
qi = np.zeros((7,1)) # Gen coordinate with euler param

# POINTS OF INTEREST, LOCAL JOINTS
uBar_FLW  = np.array([[ axleDistance/2], [-bodyHeight/2], [-axleLength/2] ]) 
uBar_FRW  = np.array([[ axleDistance/2], [-bodyHeight/2], [ axleLength/2] ])
uBar_RLW  = np.array([[-axleDistance/2], [-bodyHeight/2], [-axleLength/2] ])
uBar_RRW  = np.array([[-axleDistance/2], [-bodyHeight/2], [ axleLength/2] ])

n, nc  = 7, 1 # Generalized coordinates, number of Constraints
qi             = np.zeros((n,1))    # gen. position
qiDot          = np.zeros((n,1))    # gen. velocity
qiDotDot_lamda = np.zeros((n+nc,1)) # gen. acceleration

qi[l2i(1, "y")] = 0.3 # [m]

# Constrained equation
q_allTime       = np.zeros((np.size(time),  n))
v_allTime       = np.zeros((np.size(time),  n))
a_allTime       = np.zeros((np.size(time),  n))

def mainProg():
    global qi, qiDot, qiDotDot_lamda
    timeNow  = timeStart
    iter = 0

    for timeID in range(np.size(time)):
        max_iteration = 50
        count = 0
        delta_qi_norm = 1

        while delta_qDep_norm > epsilon:

            Cq, Cq_dep, Cq_indep, constraintVect = config(qi)
            q_dep         = np.concatenate((qi[0:2], qi[3:5]), axis = 0) # position
            q_depNew, delta_qDep_norm = conMod.positionAnalysis(constraintVect,
                                                                Cq_dep, q_dep) 
            count = count + 1
            if (delta_qDep_norm<epsilon) or (count>max_iteration):
               break
        # e. Find dependent acceleration (indep acc at the same time)
        qiDotDot_lamda = systemEquation(0, Cq, qi, qiDot)

        # f. Store everything
        q_allTime[timeID,:]      = qi.T
        v_allTime[timeID,:]      = qiDot.T
        a_allTime[timeID,:]      = qiDotDot_lamda[0:n].T

        # g. Calculate q, qdot, qdotdot independent @ t+1
        qi, qiDot = rungeKutta4_AtTimeNow( qi, qiDot, systemEquation, 
                                            stepSize, timeNow)
        iter = iter +1
        timeNow = timeNow + stepSize

    plt.figure(1)
    plt.plot(time, q_allTime[:, l2i(1, "y")])
    plt.title('y')
    plt.ylabel('position y')
    plt.xlabel('time [s]')
    plt.grid(True)
    plt.show()

# IMPORTANT CALCULATION FUNCTIONS
def config(qi): #OKAY!
    rFLW = calMod.local2global(qi, uBar_FLW, 1)
    rFRW = calMod.local2global(qi, uBar_FRW, 1)
    rRLW = calMod.local2global(qi, uBar_RLW, 1)
    rRRW = calMod.local2global(qi, uBar_RRW, 1)

    Cq = 0

    return Cq

def systemEquation(t, Cq, qi, qiDot):
    # Construct MCq matrix (MASS MODULE)

    massAugmented = np.zeros((7, 7))
    massAugmented[0:3, 0:3] = massRR
    Gbar = calMod.GBarMat(qi[l2i(1, "theta0")],qi[l2i(1, "theta1")],
                        qi[l2i(1, "theta2")], qi[l2i(1, "theta3")] )

    massAugmented[3:7, 3:7] = np.transpose(Gbar)*Itheta2*Gbar

    # Construct QeQd vector (FORCE MODULE)
    Qe = np.zeros((7,1), dtype = float)
    
    # External Force from Weight
    Qe[l2i(1, "y")] = -mass_body*gravity

    # External Force from spring
    # -joint B (link 1&2)
    QSpring1B, QSpring2B = fMod.torSpring(krB, qi, 1, 2, 0)
    # -joint C (link 2&3)
    QSpring2C, QSpring3C = fMod.torSpring(krC, qi, 2, 3, 0)

    # External Force from damper
    # -joint B (link 1&2)
    QDamp1B, QDamp2B = fMod.torDamp(crB, qiDot, 1, 2)
    # -joint C (link 2&3)
    QDamp2C, QDamp3C = fMod.torDamp(crC, qiDot, 2, 3)
    
    Qe[l2i(1,"theta")]= QSpring1B + QDamp1B
    Qe[l2i(2,"theta")]= QSpring2B + QSpring2C + QDamp2B + QDamp2C
    Qe[l2i(3,"theta")]= QSpring3C + QDamp3C

    Qd = cnMod.QdEP1(qi, 1)
        
    QeAug = np.concatenate((Qe, Qd), axis = 0) #15x1
    mass_MatInverse = np.linalg.inv(massAugmented)
    qiDotDot_lamda = np.dot(mass_MatInverse, QeAug)
    return qiDotDot_lamda

def rungeKutta4_AtTimeNow(qi, qiDot, systemFunction, stepSize, timeNow):
    # This function works with ANY number of DOF
    x = np.array([qi[l2i(1, "x")], 
                  qi[l2i(1, "y")],
                  qi[l2i(1, "z")],
                  qi[l2i(1, "theta0")],
                  qi[l2i(1, "theta1")],
                  qi[l2i(1, "theta2")],
                  qi[l2i(1, "theta3")]])

    xDot = np.array([qiDot[l2i(1, "x")], 
                     qiDot[l2i(1, "y")],
                     qiDot[l2i(1, "z")],
                     qiDot[l2i(1, "theta0")],
                     qiDot[l2i(1, "theta1")],
                     qiDot[l2i(1, "theta2")],
                     qiDot[l2i(1, "theta3")]])

    y = np.concatenate((x, xDot), axis = 0)
    numberOfDOF = int(np.size(y)/2)
    
    # RungeKutta4
    t1  = timeNow
    Cq  = config(qi)
    f_1 = systemFunction(t1, Cq, qi, qiDot)
    k1  = np.zeros((np.size(y), 1))        
    for x    in range(numberOfDOF):
        k1[x]   = y[x+numberOfDOF]
        k1[x+numberOfDOF] = f_1[l2i(x+1, "theta")]
    
    t2  = t1+0.5*stepSize
    y2  = y + 0.5*k1*stepSize
    for i in range(numberOfDOF):
        qi   [l2i(i+1, "theta")] = y2[i]
        qiDot[l2i(i+1, "theta")] = y2[i+numberOfDOF]
    Cq = config(qi)
    f_2 = systemFunction(t2, Cq, qi, qiDot)
    k2  = np.zeros((np.size(y), 1))           
    for x    in range(numberOfDOF):
        k2[x]   = y2[x+numberOfDOF]
        k2[x+numberOfDOF] = f_2[l2i(x+1, "theta")]
    
    t3  = t1+0.5*stepSize
    y3  = y + 0.5*k2*stepSize
    for i in range(numberOfDOF):
        qi   [l2i(i+1, "theta")] = y3[i]
        qiDot[l2i(i+1, "theta")] = y3[i+numberOfDOF]
    Cq  = config(qi)
    f_3 = systemFunction(t3, Cq, qi, qiDot)
    k3  = np.zeros((np.size(y), 1))           
    for x    in range(numberOfDOF):
        k3[x]   = y3[x+numberOfDOF]
        k3[x+numberOfDOF] = f_3[l2i(x+1, "theta")]
    
    t4  = t1+stepSize
    y4  = y + k3*stepSize
    for i in range(numberOfDOF):
        qi   [l2i(i+1, "theta")] = y4[i]
        qiDot[l2i(i+1, "theta")] = y4[i+numberOfDOF]
    Cq  = config(qi)
    f_4 = systemFunction(t4, Cq, qi, qiDot)
    k4  = np.zeros((np.size(y), 1))
    for x    in range(numberOfDOF):
        k4[x]   = y4[x+numberOfDOF]
        k4[x+numberOfDOF] = f_4[l2i(x+1, "theta")]

    RKFunct = (k1 + 2*k2 + 2*k3 + k4)/6

    yNew = y + stepSize*RKFunct
    for i in range(numberOfDOF):
        qi   [l2i(i+1, "theta")] = yNew[i]
        qiDot[l2i(i+1, "theta")] = yNew[i+numberOfDOF]

    return qi, qiDot

# Run main program
if __name__=="__main__":
    mainProg()