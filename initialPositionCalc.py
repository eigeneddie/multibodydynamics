#Multibody DYNAMICS CALCULATION
#import necessary packages
import numpy as np  
import pandas as pd

# USEFUL FUNCTIONS
def ATransformMatrix (theta): #A
    ATransform = np.array([[np.cos(theta), -np.sin(theta)], 
                            [np.sin(theta), np.cos(theta)]])
    return ATransform 

def ATransformMatrixTHETA (theta): #A_theta
    ATransformTHETA = np.array([[-np.sin(theta), -np.cos(theta)], 
                                [np.cos(theta), -np.sin(theta)]])
    return ATransformTHETA

# 1. KNOWN PARAMETERS
# May also be data fetched from somewhere, I dunno
lengthLink2 = 20 # [cm]
lengthLink3 = 40 # [cm]
sliderHeight = 7 # [cm]
omega2 = 2 # [rad/s]
theta2Initial = np.pi/3 # [rad]

# 2. DEFINING GENERALIZED COORDINATES AND POINTS OF INTEREST
# LOCAL COORDINATES
# Point O
u_bar_1O = np.array([[0], [0]], dtype = float)
u_bar_2O = np.array([[-lengthLink2/2], [0]], dtype = float)

# Point A
u_bar_2A = np.array([[lengthLink2/2], [0]], dtype = float) 
u_bar_3A = np.array([[-lengthLink3/2], [0]], dtype = float)

# Point B
u_bar_3B = np.array([[lengthLink3/2], [0]], dtype = float)
u_bar_4B = np.array([[0],[0]], dtype = float) 

# Generalized Coordinate vector!!

generalizedCoordinates = np.array([[0],[0],[0],[0],[0],[0],
                                [0],[0],[0],[0],[0],[0]], dtype = float)

qi = generalizedCoordinates

# 3. CONSTRAINT EQUATION
#constraintVector = np.array([np.ones(np.size(generalizedCoordinates))], dtype = float).T
constraintVector = np.array([[0],[0],[0],[0],[0],[0],
                            [0],[0],[0],[0],[0],[0]], dtype = float)
epsilon = 0.00000000000001
max_iteration = 5
count = 0
delta_qi_norm = 1

while delta_qi_norm > epsilon:

    # GLOBAL COORDINATES
    R1 = np.array([qi[0], qi[1]], dtype = float)
    R2 = np.array([qi[3], qi[4]], dtype = float)

    # Point O at link 1
    r1O = R1
    # Point O at link 2
    r2O = R2 + np.matmul(ATransformMatrix(float(qi[5])), u_bar_2O)

    # Point A at link 2
    r2A = R2 + np.matmul(ATransformMatrix(float(qi[5])), u_bar_2A)
    # Point A at link 3
    R3 = np.array([qi[6], qi[7]], dtype = float) 
    r3A = R3 + np.matmul(ATransformMatrix(float(qi[8])), u_bar_3A)

    # Point B at link 3
    r3B = R3 + np.matmul(ATransformMatrix(float(qi[8])),u_bar_3B)
    # Point B at link 4
    R4 = np.array([qi[9], qi[10]], dtype = float)
    r4B = R4 + np.matmul(ATransformMatrix(float(qi[11])),u_bar_4B)
    
    # Ground constraint
    for i in range(3):
        constraintVector[i] = qi[i]

    # Pin joint O
    constraintPinO = r1O - r2O
    for i in range(np.size(constraintPinO)):
    # Equation 4-5
        constraintVector[i+3] = constraintPinO[i]

    # Pin joint A
    constraintPinA = r2A - r3A
    for i in range(np.size(constraintPinA)):
    # Equation 6-7
        constraintVector[i+5] = constraintPinA[i]

    # Pin joint B
    constraintPinB = r3B - r4B
    for i in range(np.size(constraintPinB)):
    # Equation 8-9
        constraintVector[i+7] = constraintPinB[i]

    constraintVector[9] = qi[10]-sliderHeight
    constraintVector[10] = qi[11] - 0
    constraintVector[11] = qi[5] - theta2Initial - omega2*0 #ceritanya t = 0 dulu

    # 5. JACOBIAN MATRIX Cq
    n = np.size(qi)
    jacobianMatrixCq = np.zeros((n,n), dtype = float)
    identity3x3 = np.identity(3)
    identity2x2 = np.identity(2)
    zeros2x1 = np.zeros((2,1))

    # row 1-3
    jacobianMatrixCq[0:3,0:3] = identity3x3

    # row 4-5
    C45 = np.matmul(ATransformMatrixTHETA(float(qi[5])), u_bar_2O)
    jacobianMatrixCq[3:5,3:5] = -identity2x2
    jacobianMatrixCq[3:5,5:6] = -C45

    # row 6-7
    C67_link2 = np.matmul(ATransformMatrixTHETA(float(qi[5])), u_bar_2A)
    C67_link3 = np.matmul(ATransformMatrixTHETA(float(qi[8])), u_bar_3A)
    jacobianMatrixCq[5:7,3:5] = identity2x2
    jacobianMatrixCq[5:7,5:6] = C67_link2
    jacobianMatrixCq[5:7,6:8] = -identity2x2
    jacobianMatrixCq[5:7,8:9] = -C67_link3

    # row 8-9
    C89_link3 = np.matmul(ATransformMatrixTHETA(float(qi[8])), u_bar_3B)
    C89_link4 = np.matmul(ATransformMatrixTHETA(float(qi[11])), u_bar_4B)
    jacobianMatrixCq[7:9,6:8] = identity2x2
    jacobianMatrixCq[7:9,8:9] = C89_link3
    jacobianMatrixCq[7:9,9:11] = -identity2x2
    jacobianMatrixCq[7:9,11:12] = -C89_link4

    # row 10-11
    jacobianMatrixCq[9:11,10:12] = identity2x2

    # row 12
    jacobianMatrixCq[11][5] = 1
    jacobianMatrix_Cq = pd.DataFrame(jacobianMatrixCq, columns =
                                    ['R1x', 'R1y', 'theta1',
                                    'R2x','R2y','theta2',
                                    'R3x','R3y','theta3',
                                    'R4x','R4y','theta4'])
    
    inverse_jacobian = np.linalg.inv(jacobianMatrixCq)
    delta_qi = - np.matmul(inverse_jacobian, constraintVector)
    delta_qi_norm = np.linalg.norm(delta_qi)
    count = count + 1
    qi = qi + delta_qi

    if delta_qi_norm<epsilon or count > max_iteration:
        break

generalizezCoordinates_q = pd.DataFrame(np.transpose(qi), 
                                        columns= 
                                        ['R1x', 'R1y', 'theta1',    
                                        'R2x','R2y','theta2',
                                        'R3x','R3y','theta3',
                                        'R4x','R4y','theta4'])
                                        

print("Generalized Coordinates vector = ")
print(generalizezCoordinates_q)
print(" ")

constraintVector_C = pd.DataFrame(np.transpose(constraintVector), columns =
                                    ['C1', 'C2', 'C3',
                                    'C4','C5','C6',
                                    'C7','C8','C9',
                                    'C10','C11','C12'])


print("Constraint Vector = ")
print(constraintVector_C)
print(" ")

jacobianMatrix_Cq = pd.DataFrame(jacobianMatrixCq, columns =
                                    ['R1x', 'R1y', 'theta1',
                                    'R2x','R2y','theta2',
                                    'R3x','R3y','theta3',
                                    'R4x','R4y','theta4'])
print("Jacobian Matrix = ")
print(jacobianMatrix_Cq)
print(" ")
print("Count = ")
print(count)
print(" ")
print("That's it! The program is OVER!")