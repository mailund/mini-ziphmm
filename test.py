from __future__ import print_function

import os, sys
from time import time
import numpy as np
import mini_hmm_cython
import mini_ziphmm

# A slightly beefier HMM represeting some isolation model extracted via the
# code at github.com/mailund/IMCoalHMM/
pi = np.array([ 0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1])
T = np.array([
    [  9.99685707e-01, 3.49214287e-05, 3.49214287e-05, 3.49214287e-05, 3.49214287e-05, 3.49214287e-05, 3.49214287e-05, 3.49214287e-05, 3.49214287e-05, 3.49214287e-05],
    [  3.49214287e-05, 9.99677505e-01, 3.59467153e-05, 3.59467153e-05, 3.59467153e-05, 3.59467153e-05, 3.59467153e-05, 3.59467153e-05, 3.59467153e-05, 3.59467153e-05],
    [  3.49214287e-05, 3.59467153e-05, 9.99665674e-01, 3.76368326e-05, 3.76368326e-05, 3.76368326e-05, 3.76368326e-05, 3.76368326e-05, 3.76368326e-05, 3.76368326e-05],
    [  3.49214287e-05, 3.59467153e-05, 3.76368326e-05, 9.99649691e-01, 4.03006341e-05, 4.03006341e-05, 4.03006341e-05, 4.03006341e-05, 4.03006341e-05, 4.03006341e-05],
    [  3.49214287e-05, 3.59467153e-05, 3.76368326e-05, 4.03006341e-05, 9.99628825e-01, 4.44739426e-05, 4.44739426e-05, 4.44739426e-05, 4.44739426e-05, 4.44739426e-05],
    [  3.49214287e-05, 3.59467153e-05, 3.76368326e-05, 4.03006341e-05, 4.44739426e-05, 9.99602016e-01, 5.11760322e-05, 5.11760322e-05, 5.11760322e-05, 5.11760322e-05],
    [  3.49214287e-05, 3.59467153e-05, 3.76368326e-05, 4.03006341e-05, 4.44739426e-05, 5.11760322e-05, 9.99567689e-01, 6.26185125e-05, 6.26185125e-05, 6.26185125e-05],
    [  3.49214287e-05, 3.59467153e-05, 3.76368326e-05, 4.03006341e-05, 4.44739426e-05, 5.11760322e-05, 6.26185125e-05, 9.99523520e-01, 8.47027217e-05, 8.47027217e-05],
    [  3.49214287e-05, 3.59467153e-05, 3.76368326e-05, 4.03006341e-05, 4.44739426e-05, 5.11760322e-05, 6.26185125e-05, 8.47027217e-05, 9.99467482e-01, 1.40740829e-04],
    [  3.49214287e-05, 3.59467153e-05, 3.76368326e-05, 4.03006341e-05, 4.44739426e-05, 5.11760322e-05, 6.26185125e-05, 8.47027217e-05, 1.40740829e-04, 9.99467482e-01]])
E = np.array([
    [ 0.99795105,  0.00204895,  1.        ],
    [ 0.99784002,  0.00215998,  1.        ],
    [ 0.99771506,  0.00228494,  1.        ],
    [ 0.99757217,  0.00242783,  1.        ],
    [ 0.99740528,  0.00259472,  1.        ],
    [ 0.99720465,  0.00279535,  1.        ],
    [ 0.99695296,  0.00304704,  1.        ],
    [ 0.99661462,  0.00338538,  1.        ],
    [ 0.99609392,  0.00390608,  1.        ],
    [ 0.99471612,  0.00528388,  1.        ]])
NSYM = 3

assert E.shape[0] == T.shape[0] == T.shape[1] == pi.shape[0]
assert E.shape[1] == NSYM


# Sequence randomly generated to appear like an alignment with differences in
# about 0.25% of all positions. Not sure that makes sense but it is what the
# emission matrix above suggests.
t_0 = time()
np.random.seed(100)
weights = np.array([100,  0.25, 0.0])
weights /= weights.sum()
obs = np.array(np.random.choice(range(NSYM), 10*1000*1000, p=weights), dtype=np.int32)
sim_time = time() - t_0
print('"Simulation" time: {0}ms'.format(int(1e3*sim_time)))
print()

preprocessing = []
results = []

# Do a regular hmm forward algorithm on the observations
t_0 = time()
_, _, logL = mini_hmm_cython.calc_forward(pi, T, E, obs)
results.append(("mini-hmm", logL, time() - t_0))

# Preprocess the observations for use with mini_ziphmm
t_0 = time()
new_obs, sym2pair, new_nsyms = mini_ziphmm.preprocess_raw_observations(obs, NSYM)
preprocessing.append("mini_ziphmm     {0:9.2f}ms".format(1e3*(time() - t_0)))

# Get the likelihood with mini_ziphmm
t_0 = time()
logL = mini_ziphmm.forward(pi, T, E, sym2pair, new_obs, NSYM, new_nsyms)
results.append(("mini_ziphmm", logL, time() - t_0))

try:
    import pyZipHMM
    t_0 = time()
    i2str = [str(i) for i in range(NSYM)]
    with open("test.seq", "w") as f:
        for i in range(0, len(obs), 1000):
            f.write(" ".join(i2str[o] for o in obs[i:i+1000]))
            f.write(" ")
    t_1 = time() - t_0
    forwarder = pyZipHMM.Forwarder.fromSequence("test.seq", NSYM)
    t_2 = time() - t_0
    os.system("rm -f test.seq")
    preprocessing.append("pyZipHMM        {0:9.2f}ms".format(1e3*t_2))
    preprocessing.append(" (Initial I/O)  {0:9.2f}ms".format(1e3*t_1))
    preprocessing.append("    (pyZipHMM)  {0:9.2f}ms".format(1e3*(t_2 - t_1)))

    t_0 = time()
    zpi = pyZipHMM.Matrix(pi.shape[0], 1)
    for i in range(pi.shape[0]):
        zpi[i,0] = pi[i]
    zT = pyZipHMM.Matrix(T.shape[0], T.shape[1])
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            zT[i,j] = T[i,j]
    zE = pyZipHMM.Matrix(E.shape[0], E.shape[1])
    for i in range(E.shape[0]):
        for j in range(E.shape[1]):
            zE[i,j] = E[i,j]
    logL = forwarder.forward(zpi, zT, zE)
    results.append(("pyZipHMM", logL, time() - t_0))
except:
    pass

print("Preprocessing:")
print("\n".join(preprocessing))

print()
print("Likelihood calculations:")
for (name, L, t) in results:
    print("{0:<20} {1:15.4f} in {2:9.2f}ms".format(name, L, 1e3*t))
