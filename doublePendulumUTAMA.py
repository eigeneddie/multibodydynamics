# import necessary packages
import numpy as np  
import matplotlib.pyplot as plt
import calcModule as calMod # Important custom functions
import constraintModule as conMod
from calcModule import ATransformMatrix as ATrans, link2index, prettyMatVect2

# 1. === INPUT PARAMETERS (GLOBAL VARIABLES) ======
# ================================
link1, link2  = 1, 1 # [m]

timeStart, timeEnd, stepSize   = 0, 10, 0.01   # [s] time start, end,  increment

mass1, mass2 = 1.2, 1.2 #[kg]

# initial conditions
theta1Init = np.pi/4 # [rad]
theta2Init = np.pi/2 # [rad]

# ==============================
# POINTS OF INTEREST, LOCAL JOINTS
u_bar_1A  = np.array( [[0], [ link1/2]] )
u_bar_1B  = np.array( [[0], [-link1/2]] )
u_bar_2B  = np.array( [[0], [ link2/2]] )

# Other derived parameters
time = np.arange(timeStart, timeEnd, stepSize, dtype = float).T
inertiaJ1 = calMod.inertiaRod(mass1, link1)
inertiaJ2 = calMod.inertiaRod(mass2, link2)
massVector = np.array([[mass1], [mass1], [inertiaJ1],
                       [mass2], [mass2], [inertiaJ2]])
mass_Matrix =  calMod.massMatrix(massVector) 

gravity = 9.81 # [m/s^2]

# 3. DEFINE GENERALIZED COORDINATES - 6x1
qi       = np.zeros((6,1))# position
qiDot    = np.zeros((6,1))# velocity
qiDotDot_lamda = np.zeros((10,1))# acceleration
qi[link2index(1, "theta")] = theta1Init
qi[link2index(2, "theta")] = theta2Init

# memory variables
q_allTime       = np.zeros((np.size(time), np.size(qi)))
v_allTime       = np.zeros((np.size(time), np.size(qi)))
a_allTime       = np.zeros((np.size(time), np.size(qi)))
FReact_allTime  = np.zeros((np.size(time), 4))

# 4. CONSTRAINT EQUATION
constraintVect      = np.zeros((4,1))

# 5. Iteration
epsilon             = 0.000000000001

def main():
    global qi, qiDot, qiDotDot_lamda
    timeNow  = timeStart
    iter = 0
    # 5. Newton Rhapson
    for timeID in range(np.size(time)):
        max_iteration = 50
        count = 0
        delta_qDep_norm = 1
        # a. Find Position Dependent
        while delta_qDep_norm > epsilon:

            Cq, Cq_dep, Cq_indep, constraintVect = config(qi)
            q_dep         = np.concatenate((qi[0:2], qi[3:5]), axis = 0) # position
            q_depNew, delta_qDep_norm = conMod.positionAnalysis(constraintVect,
                                                                Cq_dep, q_dep) 
            count = count + 1
            if (delta_qDep_norm<epsilon) or (count>max_iteration):
               break

       # b. Store in qi
        qi[0:2] = q_depNew[0:2]
        qi[3:5] = q_depNew[2:4]

        # c. Find Velocity Dependent
        qDot_indep    = np.concatenate((qiDot[2:3], qiDot[5:6]), axis = 0) # velocity
        Cdi = np.dot(np.linalg.inv(-Cq_dep), Cq_indep)
        qDot_dep = np.dot(Cdi, qDot_indep)
        # d. Store in q_i_Dot
        qiDot[0:2], qiDot[3:5]= qDot_dep[0:2], qDot_dep[2:4]
        qiDot[2:3], qiDot[5:6]= qDot_indep[0:1], qDot_indep[1:2]
        # e. Find Acceleration Dependent
        qiDotDot_lamda = systemEquation(0, Cq, qi, qiDot)
        # f. Store Aceeleration on All time
        q_allTime[timeID,:] = qi.T
        v_allTime[timeID,:] = qiDot.T
        a_allTime[timeID,:] = qiDotDot_lamda[0:6].T
        FReact_allTime[timeID,:] = qiDotDot_lamda[6:10].T

        # g. Calculate q, qdot, qdotdot independent for t+1
        qi, qiDot = rungeKutta4_AtTimeStep( qi, qiDot, systemEquation, stepSize, timeNow) #DATA FOR THE NEXT TIME STEP
        iter = iter +1
        timeNow = timeNow + stepSize

    plt.figure(1)
    plt.plot(time, q_allTime[:, link2index(1, "theta")])
    plt.plot(time, q_allTime[:, link2index(2, "theta")])
    plt.title('theta1')
    plt.ylabel('rad')
    plt.xlabel('time')
    plt.grid(True)
    plt.legend(["theta 1", "theta 2"])

    plt.figure(2)
    plt.plot(time, v_allTime[:, link2index(1, "theta")])
    plt.plot(time, v_allTime[:, link2index(2, "theta")])
    plt.title('theta1')
    plt.ylabel('rad')
    plt.xlabel('time')
    plt.grid(True)
    plt.legend(["omega 1", "omega 2"])

    plt.figure(3)
    plt.plot(time, q_allTime[:, link2index(1, "x")])
    plt.plot(time, q_allTime[:, link2index(2, "x")])
    plt.title('theta1')
    plt.ylabel('rad')
    plt.xlabel('time')
    plt.grid(True)
    plt.legend(["alpha 1", "alpha 2"])
    
    plt.figure(4)
    plt.plot(time, -FReact_allTime[:, 1])
    plt.plot(time, FReact_allTime[:, 3])
    plt.title('theta1')
    plt.ylabel('rad')
    plt.xlabel('time')
    plt.grid(True)
    plt.legend(["alpha 1", "alpha 2"])
    plt.show()


def config(qi): #CLEAR
    r1A = calMod.local2global(qi, u_bar_1A, 1)
    r1B = calMod.local2global(qi, u_bar_1B, 1)
    r2B = calMod.local2global(qi, u_bar_2B, 2)
        
    # 4. CONSTRAINT EQUATION C
    constraintVect = conMod.constraintEquation(r1A, r1B, r2B) #, r3C, timeNow)

    # 5. JACOBIAN MATRIX Cq
    Cq, Cq_dep, Cq_indep = conMod.jacobianMatrix(qi, u_bar_1A, u_bar_1B, u_bar_2B)

    return Cq, Cq_dep, Cq_indep, constraintVect

def systemEquation(t, Cq, qi, qiDot):
    # Construct MCq matrix CLEAR
    massSize = mass_Matrix.shape[0]
    constVSize = constraintVect.shape[0]
    matDim =  massSize + constVSize
    massAugmented = np.zeros((matDim, matDim))
    massAugmented[0:massSize, 0:massSize] = mass_Matrix
    massAugmented[massSize:matDim, 0:massSize] = Cq
    massAugmented[0:massSize, massSize:matDim] = np.transpose(Cq)

    # Construct QeQd vector
    Qe = np.zeros((massSize,1), dtype = float)
    Qe[1] = -mass1*gravity
    Qe[4] = -mass2*gravity 

    Qd1 = conMod.QdCalc1(qi, qiDot, u_bar_1A, 1)
    Qd2 = conMod.QdCalc2(qi, qiDot, u_bar_1B, u_bar_2B, 1, 2)
    Qd = np.concatenate((-Qd1, Qd2), axis = 0)
    
    QeAug = np.concatenate((Qe, Qd), axis = 0) #10x1
    mass_MatInverse = np.linalg.inv(massAugmented)
    qiDotDot_lamda = np.dot(mass_MatInverse, QeAug)

    return qiDotDot_lamda

def rungeKutta4_AtTimeStep(qi, qiDot, systemFunction, stepSize, timeNow):
    # This function works with ANY number of DOF
    x = np.array([qi[link2index(1, "theta")], 
                qi[link2index(2, "theta")]])

    xDot = np.array([qiDot[link2index(1, "theta")], 
                    qiDot[link2index(2, "theta")]])
    y = np.concatenate((x, xDot), axis = 0)
    numberOfDOF = int(np.size(y)/2)
    
    # RungeKutta4
    t1  = timeNow
    Cq, _, _, _ = config(qi)
    f_1 = systemFunction(t1, Cq, qi, qiDot)
    k1  = np.zeros((np.size(y), 1))        
    for x    in range(numberOfDOF):
        k1[x]   = y[x+numberOfDOF]
        k1[x+numberOfDOF] = f_1[link2index(x+1, "theta")]
    
    t2  = t1+0.5*stepSize
    y2  = y + 0.5*k1*stepSize
    qi[link2index(1, "theta")], qi[link2index(2, "theta")] = y2[0], y2[1]
    qiDot[link2index(1, "theta")], qiDot[link2index(2, "theta")] = y2[2], y2[3]
    Cq, _, _, _ = config(qi)
    f_2 = systemFunction(t2, Cq, qi, qiDot)
    k2  = np.zeros((np.size(y), 1))           
    for x    in range(numberOfDOF):
        k2[x]   = y2[x+numberOfDOF]
        k2[x+numberOfDOF] = f_2[link2index(x+1, "theta")]
    
    t3  = t1+0.5*stepSize
    y3  = y + 0.5*k2*stepSize
    qi[link2index(1, "theta")], qi[link2index(2, "theta")] = y3[0], y3[1]
    qiDot[link2index(1, "theta")], qiDot[link2index(2, "theta")] = y3[2], y3[3]
    Cq, _, _, _ = config(qi)
    f_3 = systemFunction(t3, Cq, qi, qiDot)
    k3  = np.zeros((np.size(y), 1))           
    for x    in range(numberOfDOF):
        k3[x]   = y3[x+numberOfDOF]
        k3[x+numberOfDOF] = f_3[link2index(x+1, "theta")]
    
    t4  = t1+stepSize
    y4  = y + k3*stepSize
    qi[link2index(1, "theta")], qi[link2index(2, "theta")] = y4[0], y4[1]
    qiDot[link2index(1, "theta")], qiDot[link2index(2, "theta")] = y4[2], y4[3]
    Cq, _, _, _ = config(qi)
    f_4 = systemFunction(t4, Cq, qi, qiDot)
    k4  = np.zeros((np.size(y), 1))
    for x    in range(numberOfDOF):
        k4[x]   = y4[x+numberOfDOF]
        k4[x+numberOfDOF] = f_4[link2index(x+1, "theta")]

    RKFunct = (k1 + 2*k2 + 2*k3 + k4)/6

    yNew = y + stepSize*RKFunct
    qi[link2index(1, "theta")], qi[link2index(2, "theta")] = yNew[0], yNew[1]
    qiDot[link2index(1, "theta")], qiDot[link2index(2, "theta")] = yNew[2], yNew[3]

    return qi, qiDot

# Run main program
if __name__=="__main__":
    main()