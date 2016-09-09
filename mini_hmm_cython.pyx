#cython: boundscheck=False
#cython: wraparound=False
import numpy as np
cimport numpy as np
cimport cython

def calc_forward(
                 np.ndarray[double, ndim=1] pi,
                 np.ndarray[double, ndim=2] T,
                 np.ndarray[double, ndim=2] E,
                 np.ndarray[np.int32_t, ndim=1] obs):
    assert T.shape[0] == T.shape[1]
    cdef size_t k = T.shape[0]
    cdef size_t L = obs.shape[0]
    cdef np.ndarray[double, ndim=2] An = np.zeros((L, k))
    cdef np.ndarray[double, ndim=1] C = np.zeros(L)
    cdef np.ndarray[double, ndim=1] D = np.zeros(k)
    cdef np.int32_t o
    cdef double x, C_n, C_n_inv
    C_n = 0.0
    for i in xrange(k):
        C_n += pi[i] * E[i, obs[0]]
    C[0] = C_n
    for i in xrange(k):
        An[0, i] = pi[i] * E[i, obs[0]] / C_n
    cdef size_t t = 1
    # Using a range(1,L) confused cython for some reason, hence the while
    while t < L:
        o = obs[t]
        for j in xrange(k):
            x = 0.0
            for i in xrange(k):
                x += T[i, j] * An[t - 1, i]
            D[j] = x * E[j, o]
        C_n = 0.0
        for j in xrange(k):
            C_n += D[j]
        C[t] = C_n
        C_n_inv = 1.0 / C_n
        for j in xrange(k):
            An[t, j] = D[j] * C_n_inv
        t += 1
    return An, C, np.log(C).sum()


def calc_forward_backward(
                 np.ndarray[double, ndim=1] pi,
                 np.ndarray[double, ndim=2] T,
                 np.ndarray[double, ndim=2] E,
                 np.ndarray[np.int32_t, ndim=1] obs):
    assert T.shape[0] == T.shape[1]
    cdef size_t k = T.shape[0]
    cdef size_t L = obs.shape[0]
    A, C, logL = calc_forward(pi, T, E, obs)
    cdef np.ndarray[double, ndim=2] B = np.zeros((L, k))
    B[L-1, :] = 1.0
    cdef np.int32_t o
    cdef double x
    cdef size_t i, j
    cdef size_t n = L -1
    while n > 0:
        o = obs[n]
        for i in xrange(k):
            x = 0.0
            for j in xrange(k):
                x += B[n, j] * E[j, o] * T[i, j]
            B[n-1, i] = x/C[n]
        n -= 1
    return A, B, C, logL
