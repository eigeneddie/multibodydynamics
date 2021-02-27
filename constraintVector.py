import numpy as np 
import pandas as pd

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