import numpy as np
import matplotlib.pyplot as plt

maxForceSimul = np.array([5340, 8236, 7579, 7869])
maxForceExp   = np.array([4261, 7947, 8241, 8964])
maxForceError = np.absolute(np.divide(maxForceSimul-maxForceExp, maxForceExp)*100)
print(maxForceError)
print(np.average(maxForceError))

stiffnessSimul = np.array([168, 174, 167, 167, 295])
stiffnessExp   = np.array([167, 142, 298, 175, 240])
stiffnessError = np.absolute(np.divide(stiffnessSimul-stiffnessExp, stiffnessExp)*100)
print(stiffnessError)
print(np.average(stiffnessError))

caseStudy = np.array([1, 2, 3, 4])
caseStudy2 = np.array([1, 2, 3, 4,5])

plt.figure(1)
plt.plot(caseStudy, maxForceSimul, 'bo-')
plt.plot(caseStudy, maxForceExp, 'r+-')
plt.title('max force comparison')
plt.ylabel('Force [N]')
plt.xlabel('case study [#]')
plt.grid(True)
plt.legend(["Simulation", "Experiment"])

plt.figure(2)
plt.plot(caseStudy2, stiffnessSimul, 'bo-')
plt.plot(caseStudy2, stiffnessExp, 'r+-')
plt.title('Stiffness comparison')
plt.ylabel('Stiffness [N/mm]')
plt.xlabel('case study [#]')
plt.grid(True)
plt.legend(["Simulation", "Experiment"])

plt.figure(3)
plt.plot(caseStudy, maxForceError, 'bo-')
plt.title('Relative error of maximum force')
plt.ylabel('Percentage [%]')
plt.xlabel('case study [#]')
plt.grid(True)


plt.figure(4)
plt.plot(caseStudy2, stiffnessError, 'bo-')
plt.title('Relative error of stiffness')
plt.ylabel('Percentage [%]')
plt.xlabel('case study [#]')
plt.grid(True)

plt.show()

