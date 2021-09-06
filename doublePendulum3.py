# import necessary packages
import numpy as np  
import matplotlib.pyplot as plt
import calcModule as fc # Important custom functions
import constraintModule as cmod

from calcModule import ATransformMatrix as ATrans, link2index, prettyMatVect2

# 1. === INPUT PARAMETERS (GLOBAL VARIABLE) ======
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
inertiaJ1 = fc.inertiaRod(mass1, link1)
inertiaJ2 = fc.inertiaRod(mass2, link2)
massVector = np.array([[mass1], [mass1], [inertiaJ1],
                       [mass2], [mass2], [inertiaJ2]])

mass_Matrix =  fc.massMatrix(massVector) 

gravity = 9.81 # [m/s^2]
iter = 0

# 3. DEFINE GENERALIZED COORDINATES - 6x1
qi       = np.zeros((6,1))# position
qiDot    = np.zeros((6,1))# velocity
qiDotDot_lamda = np.zeros((10,1))# acceleration
qi[link2index(1, "theta")] = theta1Init
qi[link2index(2, "theta")] = theta2Init

# dependent coordinates - 4x1
q_dep         = np.concatenate((qi[0:2], qi[3:5]), axis = 0) # position
qDot_dep      = np.concatenate((qiDot[0:2], qiDot[3:5]), axis = 0) # velocity

# independent coordinates - 2x1
q_indep       = np.concatenate((qi[2:3], qi[5:6]), axis = 0) # position
qDot_indep    = np.concatenate((qiDot[2:3], qiDot[5:6]), axis = 0) # velocity

# memory variables
q_allTime       = np.zeros((np.size(time), np.size(qi)))
v_allTime       = np.zeros((np.size(time), np.size(qi)))
a_allTime       = np.zeros((np.size(time), np.size(qi)))
FReact_allTime  = np.zeros((np.size(time), 4))

# 4. CONSTRAINT EQUATION
constraintVect      = np.zeros((4,1))

# 5. Iteration
epsilon             = 0.000000000001
timeNow             = timeStart

def config(qi): #CLEAR
    r1A = fc.local2global(qi, u_bar_1A, 1)
    r1B = fc.local2global(qi, u_bar_1B, 1)
    r2B = fc.local2global(qi, u_bar_2B, 2)
        
    # 4. CONSTRAINT EQUATION C
    constraintVect = cmod.constraintEquation(r1A, r1B, r2B) #, r3C, timeNow)

    # 5. JACOBIAN MATRIX Cq
    Cq, Cq_dep, Cq_indep = cmod.jacobianMatrix(qi, u_bar_1A, u_bar_1B, u_bar_2B)

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

    Qd1 = cmod.QdCalc1(qi, qiDot, u_bar_1A, 1)
    Qd2 = cmod.QdCalc2(qi, qiDot, u_bar_1B, u_bar_2B, 1, 2)
    Qd = np.concatenate((-Qd1, Qd2), axis = 0)
    
    QeAug = np.concatenate((Qe, Qd), axis = 0) #10x1
    mass_MatInverse = np.linalg.inv(massAugmented)
    qiDotDot_lamda = np.dot(mass_MatInverse, QeAug)

    return qiDotDot_lamda

# 5. Newton Rhapson
for timeID in range(np.size(time)):
    max_iteration = 50
    count = 0
    delta_qDep_norm = 1
    
    # a. Find Position Dependent
    while delta_qDep_norm > epsilon:

        Cq, Cq_dep, Cq_indep, constraintVect = config(qi)
        q_dep         = np.concatenate((qi[0:2], qi[3:5]), axis = 0) # position

        q_depNew, delta_qDep_norm = cmod.positionAnalysis(constraintVect,
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
    #a_allTime[timeID,:] = qiDotDot_lamda[0:6].T
    #FReact_allTime[timeID,:] = qiDotDot_lamda[6:10].T

    # g. Calculate q, qdot, qdotdot independent for t+1
    qi, qiDot = rungeKutta4_timeStep( qi, qiDot, systemEquation, stepSize, timeNow) #DATA FOR THE NEXT TIME STEP

    iter = iter +1
    #print(iter)
    timeNow = timeNow + stepSize

plt.figure(1)
plt.plot(time, q_allTime[:,fc.link2index(1, "theta")])
print(q_allTime.shape[0])
print(q_allTime.shape[1])
print(np.size(q_allTime[:,fc.link2index(2, "theta")]))
print(v_allTime[0, :])
plt.title('theta1')
plt.ylabel('rad')
plt.xlabel('time')
plt.grid(True)
plt.show()