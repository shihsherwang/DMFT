from iteration import *
from utils import *
import sys
import os
wd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(wd)

g = .2
eta = .5
sigma = .1

t = 10.
Delta_t = .1
m = 1000

max_iteration = 100
damping_coefficient = .7
data_path = wd+'/'
convergence_precision = 1e-10
log_interval = 10

n = int(t/Delta_t)
C0 = 0.1 * np.eye(n)
R0 = np.eye(n, k=-1)
Cx0 = 0.1 * np.eye(n)
m0 = np.zeros(n)

parameter_model = [g, eta, sigma]
parameter_numeric = [Delta_t, n, m]
parameter_iteration = [max_iteration, damping_coefficient,
                       data_path, log_interval, convergence_precision]
initialization = [C0, R0, Cx0, m0]

DMFT = iteration(parameter_model)
DMFT.iterate(parameter_numeric, parameter_iteration, initialization)
