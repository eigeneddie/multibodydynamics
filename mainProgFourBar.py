#Multibody DYNAMICS CALCULATION
#import necessary packages
import numpy as np  
import matplotlib.pyplot as plt
import functionScript2 as fc # Important custom functions
from functionScript2 import ATransformMatrix as ATrans #Transformation matrix A_i

# FOUR BAR MECHANISM
# 1. KNOWN PARAMETERS 
link2_O2A  = 30 # [cm]
link3_AB   = 80 # [cm]
link4_O4B  = 50 # [cm]
link1_O2O4 = 60 # [cm]
omega2     = 20 # [rad/s]
theta2Initial = np.pi/2 + np.pi/6 # [rad]
C_coordinate_3x = -20 # [cm]
C_coordinate_3y = 20  # [cm]

stepTime = 0.001 # [s] time increment
timeStart = 0 # [s] start time
timeEnd = 2 # [s] end time

# Create simulation time and reserve space for vector
simulTime = np.arange(timeStart, timeEnd, stepTime, dtype = float).T

# Calculate:
# a. Point A, B, C
trajPointA = np.zeros([2, np.size(simulTime)], dtype = float).T #Point A trajectory
trajPointB = np.zeros([2, np.size(simulTime)], dtype = float).T #Point B trajectory
trajPointC = np.zeros([2, np.size(simulTime)], dtype = float).T #Point C trajectory

# b. Angular Velocity, omega [rad/s]
omega3 = np.zeros([np.size(simulTime)], dtype = float).T # omega link 3 vs time
omega4 = np.zeros([np.size(simulTime)], dtype = float).T # omega link 4 vs time

# c. Angular Acceleration, alpha [rad/s^2]
alpha3 = np.zeros([np.size(simulTime)], dtype = float).T # alpha link 3 vs time
alpha4 = np.zeros([np.size(simulTime)], dtype = float).T # alpha link 4 vs time
      
     
# 2. DEFINING GENERALIZED COORDINATES AND LOCATIONS OF POINTS OF INTEREST
# POI IN LOCAL COORDINATES 
# Point O2
u_bar_1O2 = np.array([[0], [0]], dtype = float)
u_bar_2O2 = np.array([[-link2_O2A/2], [0]], dtype = float)

# Point A
u_bar_2A = np.array([[link2_O2A/2], [0]], dtype = float) 
u_bar_3A = np.array([[-link3_AB/2], [0]], dtype = float)

# Point B
u_bar_3B = np.array([[link3_AB/2], [0]], dtype = float)
u_bar_4B = np.array([[-link4_O4B/2],[0]], dtype = float) 

# Point C 
u_bar_3C = np.array([[C_coordinate_3x], [C_coordinate_3y]], dtype = float)

# Point O4
u_bar_1O4 = np.array([[link1_O2O4], [0]], dtype = float)
u_bar_4O4 = np.array([[link4_O4B/2], [0]], dtype = float)

# Generalized Coordinate vector!!
qi = np.array([np.zeros(12)], dtype = float).T #position

# ROUGH ESTIMATES
qi[11] =  -np.pi/2                                   # theta4
qi[10] =  link4_O4B/2                                # Ry4
qi[9]  =  link1_O2O4                                 # Rx4
qi[8]  =  np.pi/20*3                                 # theta3
qi[7]  =  link4_O4B                                  # Ry3
qi[6]  =  link3_AB/2                                 # Rx3
qi[5]  =  theta2Initial                              # theta2
qi[4]  =  link2_O2A/2                                # Ry2
qi[3]  =  -link2_O2A/2*np.cos(theta2Initial-np.pi/2) #Rx2

qiDot = np.array([np.zeros(12)], dtype = float).T # velocity
qiDotDot = np.array([np.zeros(12)], dtype = float).T # acceleration

# 3. CONSTRAINT EQUATION
constraintVector = np.array([np.zeros(np.size(qi))], dtype = float).T
constraintVectorDot = np.array([np.zeros(np.size(qi))], dtype = float).T
epsilon = 0.00000000000000001
timeNow = timeStart

for timeID in range(np.size(simulTime)):
    max_iteration = 50
    count = 0
    delta_qi_norm = 1

    # FOR EVERY TIME STEPP!!
    while delta_qi_norm > epsilon:
        # Calculate GLOBAL LOCATION of Point of Interests
        r1O2, r2O2  = fc.calcGlobalCoor(qi, u_bar_1O2, u_bar_2O2, 1, 2)
        r2A, r3A    = fc.calcGlobalCoor(qi, u_bar_2A , u_bar_3A , 2, 3)
        r3B, r4B    = fc.calcGlobalCoor(qi, u_bar_3B , u_bar_4B , 3, 4)
        r1O4, r4O4  = fc.calcGlobalCoor(qi, u_bar_1O4, u_bar_4O4, 1, 4)                               
        
        # 4. CONSTRAINT EQUATION C
        constraintVector = fc.constraintEquation(constraintVector, qi, r1O2, r2O2, 
                                            r2A, r3A, r3B, r4B, r1O4, r4O4,
                                            theta2Initial, omega2, timeNow)

        constraintVectorDot = fc.constrEqDot(constraintVectorDot, omega2)

        # 5. JACOBIAN MATRIX Cq
        jacobianMatrixCq = fc.jacobianMatrix(qi, u_bar_2O2, u_bar_2A, u_bar_3A, 
                                            u_bar_3B, u_bar_4B, u_bar_1O4, u_bar_4O4)
        
        # 6. POSITION ANALYSIS
        qi, delta_qi_norm = fc.positionAnalysis(constraintVector,
                                                jacobianMatrixCq, qi)

        # 7. Velocity ANLAYSIS
        qiDot = fc.velocityAnalysis(constraintVectorDot, jacobianMatrixCq, qiDot)

        # 8. ACCELERATION ANALYSIS
        qiDotDot = fc.accelerationAnalysis(jacobianMatrixCq, qiDotDot, qiDot, qi,
                                u_bar_2O2, u_bar_2A, u_bar_3A, u_bar_3B, u_bar_4B,
                                u_bar_1O4, u_bar_4O4)
                                
        count = count + 1
        
        if (delta_qi_norm<epsilon) or (count>max_iteration):
            break

    # Trajectory point A, B, C
    r3A, r3B = fc.calcGlobalCoor(qi, u_bar_3A, u_bar_3B, 3, 3)
    r3C, r4O4 = fc.calcGlobalCoor(qi, u_bar_3C, u_bar_4O4, 3, 4)

    #
    trajPointA[timeID] = r3A.T
    trajPointB[timeID] = r3B.T
    trajPointC[timeID] = r3C.T

    # Angular velocity and acceleration Link 3
    omega3[timeID] = qiDot[8] #qi[6]#
    alpha3[timeID] = qiDotDot[8] #qi[7]#

    # Angular velocity and acceleration Link 4
    omega4[timeID] = qiDot[11]
    alpha4[timeID] = qiDotDot[11]

    timeNow = timeNow + stepTime

jacobianMatrix_Cq = fc.prettyMatVect(jacobianMatrixCq)
print("Jacobian Matrix = ")
print(jacobianMatrix_Cq)
print(" ")

constraintVector_C = fc.prettyMatVect(np.transpose(constraintVector))
print("Constraint Vector = ")
print(constraintVector_C)
print(" ")
print("sanity check distance point B and R4")
#print((r3B-R4)) # harus 25 cm (and it is!!!)

plt.figure(1)
plt.plot(trajPointA[:,0:1], trajPointA[:,1:2])
plt.title('Trajectory Point A')
plt.ylabel('Position y [cm]')
plt.xlabel('Position X [cm]')
plt.grid(True)


plt.figure(2)
plt.plot(trajPointB[:,0:1], trajPointB[:,1:2])
plt.title('Trajectory Point B')
plt.ylabel('Position y [cm]')
plt.xlabel('Position X [cm]')
plt.grid(True)


plt.figure(3)
plt.plot(trajPointC[:,0:1], trajPointC[:,1:2])
plt.title('Trajectory Point C')
plt.ylabel('Position y [cm]')
plt.xlabel('Position X [cm]')
plt.grid(True)

plt.figure(4)
plt.plot(simulTime, omega3)
plt.title('LINK 3 ANGULAR Velocity')
plt.ylabel('Omega3 [rad/s^2]')
plt.xlabel('time [s]')
plt.grid(True)


plt.figure(5)
plt.plot(simulTime, alpha3)
plt.title('LINK 3 ANGULAR ACCELERATION')
plt.ylabel('Alpha3 [rad/s^2]')
plt.xlabel('time [s]')
plt.grid(True)


plt.figure(6)
plt.plot(simulTime, omega4)
plt.title('LINK 4 ANGULAR Velocity')
plt.ylabel('Omega4 [rad/s^2]')
plt.xlabel('time [s]')
plt.grid(True)


plt.figure(7)
plt.plot(simulTime, alpha4)
plt.title('LINK 4 ANGULAR ACCELERATION')
plt.ylabel('Alpha4 [rad/s^2]')
plt.xlabel('time [s]')
plt.grid(True)
plt.show()

'''generalizezCoordinates_q = fc.prettyMatVect(np.transpose(qi))
print("Generalized Coordinates vector = ")
print(generalizezCoordinates_q)
print(" ")

constraintVector_C = fc.prettyMatVect(np.transpose(constraintVector))
print("Constraint Vector = ")
print(constraintVector_C)
print(" ")

constraintVectorDot_Ct = fc.prettyMatVect(np.transpose(constraintVectorDot))
print("Constraint Vector (velocity) = ")
print(constraintVectorDot_Ct)
print(" ")

jacobianMatrix_Cq = fc.prettyMatVect(jacobianMatrixCq)
print("Jacobian Matrix = ")
print(jacobianMatrix_Cq)
print(" ")

inverseJacob = fc.prettyMatVect(np.linalg.inv(jacobianMatrixCq))
print("Inverse Jacobian Matrix = ")
print(inverseJacob)
print(" ")

print("delta qi norm = ")
print(delta_qi_norm)
print(" ")

print("iteration count = ")
print(count)
print(" ")'''
