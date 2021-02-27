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

def replace_submatrix(mat, ind1, ind2, mat_replace):
  for i, index in enumerate(ind1):
    mat[index, ind2] = mat_replace[i, :]
  return mat

# 1. KNOWN PARAMETERS
# May also be data fetched from somewhere, I dunno
lengthLink2 = 20 # [cm]
lengthLink3 = 40 # [cm]
sliderHeight = 7 # [cm]
omega2 = 2 # [rad/s]
theta2Initial = np.pi/3 # [rad]

# Calculate initial conditions
sinTheta3 = (lengthLink2*np.sin(theta2Initial)-sliderHeight)/lengthLink3 #dummy variable
minusTheta3 = np.arcsin(sinTheta3) 
theta3Initial =2*np.pi-minusTheta3 # [rad] 
slider_X_B = lengthLink2*np.cos(theta2Initial)+lengthLink3*np.cos(theta3Initial) # [cm]
print(theta3Initial)
print(np.sin(theta3Initial))
# 2. DEFINING GENERALIZED COORDINATES
R1x, R1y, theta1  = 0, 0, 0 # this is final!!!
R2x, R2y, theta2  = 0, 0, theta2Initial
R3x, R3y, theta3  = 0, 0, theta3Initial
R4x, R4y, theta4  = slider_X_B, sliderHeight, 0
# ini sementara 

# 3. POINTS OF INTEREST
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

# GLOBAL COORDINATES
R2 = np.array([[lengthLink2/2*np.cos(theta2)],
                [lengthLink2/2*np.sin(theta2)]], dtype = float)
R1 = np.array([[R1x], [R1y]], dtype = float)
# Point O at link 1
r1O = R1
# Point O at link 2
r2O = R2 + np.matmul(ATransformMatrix(theta2),u_bar_2O)

# Point A at link 2
r2A = R2 + np.matmul(ATransformMatrix(theta2),u_bar_2A)
# Point A at link 3
R3 = r2A + np.array([[lengthLink3/2*np.cos(theta3)], 
                    [lengthLink3/2*np.sin(theta3)]], dtype = float) 
r3A = R3 + np.matmul(ATransformMatrix(theta3),u_bar_3A)

# Point B at link 3
r3B = R3 + np.matmul(ATransformMatrix(theta3),u_bar_3B)
# Point B at link 4
R4 = np.array([[R4x], [R4y]], dtype = float)
r4B = R4 + np.matmul(ATransformMatrix(theta4),u_bar_4B)

# Generalized Coordinate vector!!
'''generalizedCoordinates = np.array([[float(r1O[0])], [float(r1O[1])], [float(theta1)], 
                        [float(R2[0])], [float(R2[1])], [float(theta2)], 
                        [float(R3[0])], [float(R3[1])], [float(theta3)], 
                        [float(R4[0])], [float(R4[1])], [float(theta4)]])'''

generalizedCoordinates = np.array([[r1O[0]], [r1O[1]], [theta1], 
                        [R2[0]], [R2[1]], [theta2], 
                        [R3[0]], [R3[1]], [theta3], 
                        [R4[0]], [R4[1]], [theta4]], dtype = float)
                        
generalizezCoordinates_q = pd.DataFrame(np.transpose(generalizedCoordinates), 
                                        columns= 
                                        ['R1x', 'R1y', 'theta1',    
                                        'R2x','R2y','theta2',
                                        'R3x','R3y','theta3',
                                        'R4x','R4y','theta4'])
print(" ")
print("Generalized Coordinates vector = ")
print(generalizezCoordinates_q)
print(" ")

# 4. CONSTRAINT EQUATION

constraintVector = np.array([np.empty(np.size(generalizedCoordinates))], dtype = float).T

# Ground constraint
for i in range(3):
    constraintVector[i] = generalizedCoordinates[i]

# Pin joint O
constraintPinO = r2O - r1O
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

constraintVector[9] = r4B[1]-sliderHeight
constraintVector[10] = theta4 - 0
constraintVector[11] = theta2 - theta2Initial - omega2*0 #ceritanya t = 0 dulu

constraintVector_C = pd.DataFrame(np.transpose(generalizedCoordinates), columns =
                                    ['C1', 'C2', 'C3',
                                    'C4','C5','C6',
                                    'C7','C8','C9',
                                    'C10','C11','C12'])


print("Constraint Vector = ")
print(constraintVector_C)
print(" ")

# 5. JACOBIAN MATRIX Cq
n = np.size(generalizedCoordinates)
jacobianMatrixCq = np.zeros((n,n))
identity3x3 = np.identity(3)
identity2x2 = np.identity(2)
zeros2x1 = np.zeros((2,1))

# row 1-3
jacobianMatrixCq[0:3,0:3] = identity3x3

# row 4-5
C45 = np.matmul(ATransformMatrixTHETA(theta2), u_bar_2O)
jacobianMatrixCq[3:5,3:5] = identity2x2
jacobianMatrixCq[3:5,5:6] = C45

# row 6-7
C67_link2 = np.matmul(ATransformMatrixTHETA(theta2), u_bar_2A)
C67_link3 = np.matmul(ATransformMatrixTHETA(theta3), u_bar_3A)
jacobianMatrixCq[5:7,3:5] = identity2x2
jacobianMatrixCq[5:7,5:6] = C67_link2
jacobianMatrixCq[5:7,6:8] = -identity2x2
jacobianMatrixCq[5:7,8:9] = C67_link3

# row 8-9
C89_link3 = np.matmul(ATransformMatrixTHETA(theta3), u_bar_3B)
C89_link4 = np.matmul(ATransformMatrixTHETA(theta4), u_bar_4B)
jacobianMatrixCq[7:9,6:8] = identity2x2
jacobianMatrixCq[7:9,8:9] = C89_link3
jacobianMatrixCq[7:9,9:11] = -identity2x2
jacobianMatrixCq[7:9,11:12] = C89_link4

# row 10-11
jacobianMatrixCq[9:11,10:12] = identity2x2

# row 12
jacobianMatrixCq[11][5] = 1
jacobianMatrix_Cq = pd.DataFrame(jacobianMatrixCq, columns =
                                ['R1x', 'R1y', 'theta1',
                                'R2x','R2y','theta2',
                                'R3x','R3y','theta3',
                                'R4x','R4y','theta4'])

                            
print("Jacobian Matrix = ")
print(jacobianMatrix_Cq)
print(" ")
print(" ")
#print( jacobianMatrixCq[[0,2],:], [:,[0,2]]]   )
# 6. Successful!!!
inverse_jacob = np.linalg.inv(jacobianMatrixCq)
print(inverse_jacob)
print("EVALUATION:") 
print("Fail to figure out concatination of matricesi in Python")
print("Haven't figured out how to append matrices in Python!!!")
print("WILL DO BETTER NEXT TIME!!!")



