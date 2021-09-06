import numpy as np

def rungeKutta4 (y, time, systemFunction, stepSize):
    # This function works with ANY number of DOF

    numberOfDOF = int(np.size(y)/2)
    totalData   = np.size(time)
    state       = np.zeros((totalData,numberOfDOF), dtype = float)
    stateDot    = np.zeros((totalData,numberOfDOF), dtype = float)
    stateDotDot = np.zeros((totalData,numberOfDOF), dtype = float)
    
    # RungeKutta4
    for currentTime in range(totalData):
        
        t1  = time[currentTime]
        f_1 = systemFunction(y, t1)
        k1  = np.zeros((np.size(y), 1), dtype = float)        
        for x    in range(numberOfDOF):
            k1[x]   = y[x+numberOfDOF]
            k1[x+numberOfDOF] = f_1[x]

        t2  = time[currentTime]+0.5*stepSize
        y2  = y + 0.5*k1*stepSize
        f_2 = systemFunction(y2, t2)
        k2  = np.zeros((np.size(y), 1), dtype = float)           
        for x    in range(numberOfDOF):
            k2[x]   = y2[x+numberOfDOF]
            k2[x+numberOfDOF] = f_2[x]
        
        t3  = time[currentTime]+0.5*stepSize
        y3  = y + 0.5*k2*stepSize
        f_3 = systemFunction(y3, t3)
        k3  = np.zeros((np.size(y), 1), dtype = float)           
        for x    in range(numberOfDOF):
            k3[x]   = y3[x+numberOfDOF]
            k3[x+numberOfDOF] = f_3[x]

        t4  = time[currentTime]+stepSize
        y4  = y + k3*stepSize
        f_4 = systemFunction(y4, t4)
        k4  = np.zeros((np.size(y), 1), dtype = float)
        for x    in range(numberOfDOF):
            k4[x]   = y4[x+numberOfDOF]
            k4[x+numberOfDOF] = f_4[x]

        RKFunct = (k1 + 2*k2 + 2*k3 + k4)/6

        y = y + stepSize*RKFunct
        
        for dof in range(numberOfDOF):
            state       [currentTime][dof] = y[dof]
            stateDot    [currentTime][dof] = y[dof+numberOfDOF]
            stateDotDot [currentTime][dof] = RKFunct[dof+numberOfDOF]
       
    return state, stateDot, stateDotDot

