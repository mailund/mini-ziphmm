import os, sys
from time import time
from math import *
import numpy as np
import mini_hmm_weave
import mini_hmm_cython
import mini_ziphmm

NSYM = 3

T = np.array([[0.25, 0.75],
              [0.75, 0.25]])

E = np.array([[0.9, 0.1, 0.1],
              [0.1, 0.1, 0.9]])

pi = np.array([0.5, 0.5])

# 1112111112
# obs = np.array(map(int, "100100222112"), dtype=np.int32)
# obs = np.array(map(int, "1001"), dtype=np.int32)
np.random.seed(100)
obs = np.array(np.random.choice(range(NSYM), 1000), dtype=np.int32)

# weave vs cython
# A, B, C, logL = mini_hmm_weave.calc_forward_backward(pi, T, E, obs)
# A2, B2, C2, logL2 = mini_hmm_cython.calc_forward_backward(pi, T, E, obs)
# print "A:", np.isclose(A, A2).all()
# print "B:", np.isclose(B, B2).all()
# print "C:", np.isclose(C, C2).all()

# Disregard the first time where it compiles/loads the C code
for _ in xrange(2):
    t_weave = time()
    A, C, logL = mini_hmm_weave.calc_forward(pi, T, E, obs)
    t_weave = time() - t_weave

t_cython = time()
A2, C2, logL2 = mini_hmm_cython.calc_forward(pi, T, E, obs)
t_cython = time() - t_cython

t_zip_simple = time()
logL3 = mini_ziphmm.simple_forward(pi, T, E, obs)
t_zip_simple = time() - t_zip_simple

t_zip_preprocess = time()
new_obs, sym2pair, new_nsyms = mini_ziphmm.preprocess_raw_observations(obs, NSYM)
t_zip_preprocess = time() - t_zip_preprocess
print("zip preprocessing time: {0:0.2f}ms".format(1e3*t_zip_preprocess))

t_zip = time()
logL4 = mini_ziphmm.forward(pi, T, E, sym2pair, new_obs, NSYM, new_nsyms)
t_zip = time() - t_zip

results = [
        ("weave",       logL,   t_weave),
        ("cython",      logL2,  t_cython),
        ("zip-simple",  logL3,  t_zip_simple),
        ("zip",         logL4,  t_zip),
        ]

try:
    import pyZipHMM
    t_ziphmm_preprocess = time()
    with open("test.seq", "w") as f:
        for o in obs:
            print >>f, o
    forwarder = pyZipHMM.Forwarder.fromSequence("test.seq", NSYM)
    # forwarder.writeToDirectory("zipdir")
    #os.system("echo -n 'pyZipHMM obs: '; cat zipdir/nStates2seq/2.seq")
    #print
    os.system("rm -rf zipdir")
    os.system("rm -f test.seq")
    t_ziphmm_preprocess = time() - t_ziphmm_preprocess
    t_ziphmm = time()
    zpi = pyZipHMM.Matrix(pi.shape[0], 1)
    for i in xrange(pi.shape[0]):
        zpi[i,0] = pi[i]
    zT = pyZipHMM.Matrix(T.shape[0], T.shape[1])
    for i in xrange(T.shape[0]):
        for j in xrange(T.shape[1]):
            zT[i,j] = T[i,j]
    zE = pyZipHMM.Matrix(E.shape[0], E.shape[1])
    for i in xrange(E.shape[0]):
        for j in xrange(E.shape[1]):
            zE[i,j] = E[i,j]
    logL5 = forwarder.forward(zpi, zT, zE)
    t_ziphmm = time() - t_ziphmm
    results.append(("pyZipHMM", logL5, t_ziphmm))
    print("pyZipHMM preprocessing time: {0:0.2f}ms".format(1e3*t_ziphmm_preprocess))
except:
    pass

for (name, L, t) in results:
    print "{0:<20} {1:15} in {2:0.2f}ms".format(name, L, 1e3*t)
