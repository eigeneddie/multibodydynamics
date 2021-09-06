from functionScript2 import eulerMethod
import numpy as np 
import matplotlib.pyplot as plt

# System Parameters
#================
x_init = 0.5
xDot_init = 0

mass = 5
damp = 10
spring = 50

omega = 2
forceMagnit = 0

timeStart = 0
timeStop = 10
stepTime = 0.001
#=================

time = np.arange(timeStart, timeStop, stepTime, dtype = float)

# Initial Condition of state
y = np.array([[x_init], [xDot_init]], dtype = float)
systemFunction = lambda t, y: (forceMagnit*np.sin(omega*t)-damp*float(y[1])-spring*float(y[0]))/mass

# Using Euler Method
pos, vel, accel = eulerMethod(y, time, stepTime, systemFunction)

# Usign Runge Kutta 4-5
posRK, velRK, accelRK = eulerMethod(y, time, stepTime, systemFunction)

plt.figure(1)
plt.plot(time, pos)
plt.plot(time, posRK)
title = "position plot [step time = %1.6f s]" % stepTime
plt.title(title)
plt.ylabel('displacement [m]')
plt.xlabel('time [s]')
plt.grid(True)
plt.legend(["euler", "RK"])


plt.figure(2)
plt.plot(time, vel)
plt.plot(time, velRK)
title = "velocity plot [step time = %1.6f s]" % stepTime
plt.title(title)
plt.ylabel('velocity [m/s]')
plt.xlabel('time [s]')
plt.grid(True)
plt.legend(["euler", "RK"])


plt.figure(3)
plt.plot(time, accel)
plt.plot(time, accelRK)
title = "Accleration plot [step time = %1.6f s]" % stepTime
plt.title(title)
plt.ylabel('Acceleration [m/s/s]')
plt.xlabel('time [s]')
plt.grid(True)
plt.legend(["euler", "RK"])
plt.show()


'''
t1 = time[i]
f_1 = systemFunction(y, t1)
k1 = np.array([y[2], y[3], 
            f_1[0], f_1[1]], dtype = float)
t2 = time[i]+0.5*stepSize
y2 = y + 0.5*k1*stepSize
f_2 = systemFunction(y2, t2)
k2 = np.array([y2[2], y2[3], 
            f_2[0], f_2[1]], dtype = float)
t3 = time[i]+0.5*stepSize
y3 = y + 0.5*k2*stepSize
f_3 = systemFunction(y3, t3)
k3 = np.array([y3[2], y3[3], 
            f_3[0], f_3[1]], dtype = float)
t4 = time[i]+stepSize
y4 = y + k3*stepSize
f_4 = systemFunction(y4, t4)
k4 = np.array([y4[2], y4[3], 
            f_4[0], f_4[1]], dtype = float) #4x1
RKFunct = (k1 + 2*k2 + 2*k3 + k4)/6
'''
'''
y = y + stepSize*RKFunct   
state[i] = [float(y[0]), float(y[1])]
stateDot[i] = [float(y[2]), float(y[3])]
stateDotDot[i] = [float(RKFunct[2]), float(RKFunct[3])]
'''




