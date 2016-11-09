import os
import scipy as sp
from scipy.stats import gamma
import matplotlib.pyplot as plt

sp.random.seed(3)

x = sp.arange(1, 31*24)
y = sp.array(200*(sp.sin(2*sp.pi*x/(7*24))), dtype=int)



print(x)
print(y)
