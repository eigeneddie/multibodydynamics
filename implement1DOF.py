# 1 DOF SYSTEM
import numpy as np
import matplotlib.pyplot as plt
import libraryTugas as lib

# 1. SYSTEMS PARAMETERS
#================
# a. Initial condition
x_init1, xDot_init1 =  0.1, 0   # [m], [m/s]

# b. System parameters
mass1, damp1, spring1 = 1, 1, 10 # [kg], [Ns/m], [N/m]
omega, forceMagnit = 2, 0.5 # [rad/s], [Newton]--> SYSTEM INPUT

# c. Time parameters
timeStart, timeStop, stepTime = 0, 30, 0.001 # [S]

# d. Define System MODEL!
def systemFunction (y, t):
    xDotDot = np.zeros((1,1), dtype = float) 
    xDotDot[0] = (forceMagnit*np.sin(omega*t)-damp1*float(y[1])-spring1*float(y[0]))/mass1
    return xDotDot #*np.sin(omega*t)

# 2. INITIALIZE SIMULATION
# a. Simulation time
time = np.arange(timeStart, timeStop, stepTime, dtype = float) 

# b. Initial Condition of THE state
y = np.array([[x_init1], [xDot_init1]], dtype = float)

# 3. SOLVING EQUATION OF MOTION USING RUNGE KUTTA 4th ORDER!!
position, velocity, acceleration = lib.rungeKutta4(y, time, 
                                                systemFunction, stepTime)

# 4. PLOTING RESULTS!!
plt.figure(1)
plt.plot(time, position)
title = "position plot [step time = %1.6f s]" % stepTime
plt.title(title)
plt.ylabel('displacement [m]')
plt.xlabel('time [s]')
plt.grid(True)

plt.figure(2)
plt.plot(time, velocity)
title = "velocity plot [step time = %1.6f s]" % stepTime
plt.title(title)
plt.ylabel('velocity [m/s]')
plt.xlabel('time [s]')
plt.grid(True)

plt.figure(3)
plt.plot(time, acceleration)
title = "acceleration plot [step time = %1.6f s]" % stepTime
plt.title(title)
plt.ylabel('acceleration [m/s/s]')
plt.xlabel('time [s]')
plt.grid(True)
plt.show()
