# 2 DOF SYSTEM
import numpy as np
import matplotlib.pyplot as plt
import libraryTugas as lib

# 1. SYSTEMS PARAMETERS
#================
# a. Initial condition Block 1 & 2
x_init1, xDot_init1 =  0.1, 0   # [m], [m/s]
x_init2, xDot_init2 = -0.1, 0   # [m], [m/s]

# b. System parameters Block 1 & 2
mass1, damp1, spring1 = 1, 1, 10 # [kg], [Ns/m], [N/m]
mass2, damp2, spring2 = 2, 1, 20 # [kg], [Ns/m], [N/m]

# c. Time parameters
timeStart, timeStop, stepTime = 0, 10, 0.001 # [S]

# d. Define System MODEL!
def systemFunction (y, t): # Fungsi persamaan gerak
    Qe1 = -spring1*float(y[0]) + spring2*(float(y[1])-float(y[0]))- damp1*float(y[2]) + damp2*(float(y[3])-float(y[2]))
    Qe2 = -spring2*(float(y[1])-float(y[0])) - damp2*(float(y[3])-float(y[2]))
    externalForces_Matrix = np.array([[Qe1], [Qe2]], dtype = float)
    xDotDot = np.dot(mass_MatInverse, externalForces_Matrix)
    # dalam kasus ini, parameter "t" tidak terpakai
    return xDotDot

#======================

# 2. INITIALIZE SIMULATION
# a. Simulation time
time = np.arange(timeStart, timeStop, stepTime, dtype = float) 
# b. Mass matrix
mass_Matrix = np.array([[mass1, 0], 
                        [0, mass2]], dtype = float)                   
mass_MatInverse = np.linalg.inv(mass_Matrix)

# c. Initial Condition of THE state
y = np.array([[x_init1], [x_init2], [xDot_init1], [xDot_init2]], dtype = float)

# 3. SOLVING EQUATION OF MOTION USING RUNGE KUTTA 4th ORDER!!
position, velocity, acceleration = lib.rungeKutta4(y, time, 
                                                systemFunction, stepTime)

# 4. PLOTING RESULTS!!
plt.figure(1)
plt.plot(time, position[:,0:1])
plt.plot(time, position[:,1:2])
title = "position plot [step time = %1.6f s]" % stepTime
plt.title(title)
plt.ylabel('displacement [m]')
plt.xlabel('time [s]')
plt.grid(True)
plt.legend(["mass 1", "mass 2"])

plt.figure(2)
plt.plot(time, velocity[:,0:1])
plt.plot(time, velocity[:,1:2])
title = "velocity plot [step time = %1.6f s]" % stepTime
plt.title(title)
plt.ylabel('velocity [m/s]')
plt.xlabel('time [s]')
plt.grid(True)
plt.legend(["mass 1", "mass 2"])

plt.figure(3)
plt.plot(time, acceleration[:,0:1])
plt.plot(time, acceleration[:,1:2])
title = "acceleration plot [step time = %1.6f s]" % stepTime
plt.title(title)
plt.ylabel('acceleration [m/s/s]')
plt.xlabel('time [s]')
plt.grid(True)
plt.legend(["mass 1", "mass 2"])
plt.show()