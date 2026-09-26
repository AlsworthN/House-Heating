import matplotlib.pyplot as plt
import numpy as np
import math

T0 = 20
Tout = 10
k = 0.1
dt = 1
hours = 24

# T(t) = Tout + (T0 - Tout)*e^(-kt)

times = []
temps = []

for t in range(0,hours):
    temp = Tout + (T0 - Tout)*math.e**(-k*t)
    times.append(t)
    temps.append(temp)

plt.plot(times,temps)
plt.show()

print(temps)