import numpy as np 
import matplotlib.pyplot as plt

# System Parameters
x_init = 0
xDot_init = 0

mass = 5
damp = 10
spring = 50

omega = 2
forceMagnit = 1

timeStart = 0
timeStop = 10
stepTime = 0.001
time = np.arange(timeStart, timeStop, stepTime, dtype = float)

# Initial Condition
y = np.array([[x_init], [xDot_init]], dtype = float)


# Reserve data
totalData = np.size(time)
position = np.zeros(totalData, dtype = float)
velocity = np.zeros(totalData, dtype = float)
acceleration = np.zeros(totalData, dtype = float)

positionRK = np.zeros(totalData, dtype = float)
velocityRK = np.zeros(totalData, dtype = float)
accelerationRK = np.zeros(totalData, dtype = float)

#Euler Method
systemFunction = lambda t, y: (forceMagnit*np.sin(omega*t)-damp*float(y[1])-spring*float(y[0]))/mass
#
for i in range(totalData):
    eulerMeth = systemFunction(time[i], y)
    y = y + stepTime*np.array([y[1], [eulerMeth]], dtype = float)

    position[i] = (y[0])
    velocity[i] = (y[1])
    acceleration[i] = eulerMeth


#Runge Kutta 4-5
#systemFunction = lambda t, y: (forceMagnit*np.sin(omega*t)-damp*float(y[1])-spring*float(y[0]))/mass
y = np.array([[x_init], [xDot_init]], dtype = float)

for i in range(totalData):
    t1 = time[i]
    f_1 = systemFunction(t1, y)
    k1 = np.array([y[1], [f_1]], dtype = float)
    
    t2 = time[i]+0.5*stepTime
    y2 = y + 0.5*k1*stepTime
    f_2 = systemFunction(t2, y2)
    k2 = np.array([y2[1], [f_2]], dtype = float)

    t3 = time[i]+0.5*stepTime
    y3 = y + 0.5*k2*stepTime
    f_3 = systemFunction(t3, y3)
    k3 = np.array([y3[1], [f_3]], dtype = float)

    t4 = time[i]+stepTime
    y4 = y + k3*stepTime
    f_4 = systemFunction(t4, y4)
    k4 = np.array([y4[1], [f_4]], dtype = float)

    RKFunct = (k1 + 2*k2 + 2*k3 + k4)/6
    y = y + stepTime*RKFunct

    positionRK[i] = (y[0])
    velocityRK[i] = (y[1])
    accelerationRK[i] = RKFunct[1]


plt.figure(1)
plt.plot(time, position)
plt.plot(time, positionRK)
title = "position plot [step time = %1.6f s]" % stepTime
plt.title(title)
plt.ylabel('displacement [m]')
plt.xlabel('time [s]')
plt.grid(True)
plt.legend(["euler", "RK"])


plt.figure(2)
plt.plot(time, velocity)
plt.plot(time, velocityRK)
title = "velocity plot [step time = %1.6f s]" % stepTime
plt.title(title)
plt.ylabel('velocity [m/s]')
plt.xlabel('time [s]')
plt.grid(True)
plt.legend(["euler", "RK"])


plt.figure(3)
plt.plot(time, acceleration)
plt.plot(time, accelerationRK)
title = "Accleration plot [step time = %1.6f s]" % stepTime
plt.title(title)
plt.ylabel('Acceleration [m/s/s]')
plt.xlabel('time [s]')
plt.grid(True)
plt.legend(["euler", "RK"])
plt.show()





