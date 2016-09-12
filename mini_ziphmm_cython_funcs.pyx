# cython: boundscheck=False, wraparound=False
from __future__ import division

import numpy as np

cimport cython
cimport numpy as np
cimport libc.math


def zip_forward(
        np.ndarray[double, ndim=1] pi,
        np.ndarray[double, ndim=2] T,
        np.ndarray[double, ndim=2] E,
        np.ndarray[np.int32_t, ndim=1] obs,
        np.ndarray[double, ndim=3] sym2mat,
        np.ndarray[double, ndim=1] sym2scale):
    cdef:
        np.ndarray[double, ndim=1] tmp, res
        np.int32_t o
        size_t i, j, k, N, obs_len
        double logL, s, res_scale
    o = obs[0]
    res = pi * E[:, o]
    tmp = np.zeros_like(res)
    res_scale = res.sum()
    assert sym2mat.shape[1] == sym2mat.shape[2] == T.shape[0] == T.shape[1]
    N = T.shape[0]
    obs_len = len(obs)
    res /= res_scale
    logL = libc.math.log(res_scale)
    for i in range(1, obs_len):
        o = obs[i]
        res_scale = 0.0
        for j in range(N):
            s = 0.0
            for k in range(N):
                s += sym2mat[o, j, k] * res[k]
            tmp[j] = s
            res_scale += s
        logL += libc.math.log(res_scale)
        logL += sym2scale[o]
        res_scale = 1.0 / res_scale
        for j in range(N):
            res[j] = tmp[j] * res_scale
    return logL


def preprocess_raw_observations(
        np.ndarray[np.int32_t, ndim=1] obs,
        np.int32_t nsyms):
    cdef:
        np.int32_t counted_last_pair, p_prev, p, new_sym
        np.int32_t max_pair, max_pair_left, max_pair_right, max_pair_count
        size_t idx, i, j
        size_t obs_len
        np.int32_t prev_max_count
        np.ndarray[np.int32_t, ndim=1] pair_counts, new_obs
    sym2pair = dict()
    prev_max_count = -1
    obs_len = len(obs)
    while True:
        pair_counts = np.zeros(nsyms**2, dtype=np.int32)
        counted_last_pair = 1
        idx = 1
        while idx < obs_len - 1:
            p_prev = obs[idx-1] * nsyms + obs[idx]
            p = obs[idx] * nsyms + obs[idx+1]
            if counted_last_pair == 0 or p_prev != p:
                pair_counts[p] += 1
                counted_last_pair = 1
            else:
                counted_last_pair = 0
            idx += 1
        max_pair = pair_counts.argmax()
        max_pair_count = pair_counts[max_pair]
        if max_pair_count == prev_max_count:
            break
        else:
            prev_max_count = max_pair_count
        max_pair_left = max_pair // nsyms
        max_pair_right = max_pair - max_pair_left * nsyms
        new_sym = nsyms
        sym2pair[new_sym] = (max_pair_left, max_pair_right)
        # Replace all occurences of max_pair in obs with new_sym
        new_obs = np.zeros_like(obs)
        i, j = 1, 1
        new_obs[0] = obs[0]
        while i < obs_len:
            if i + 1 < obs_len and max_pair == obs[i]*nsyms + obs[i+1]:
                new_obs[j] = new_sym
                i += 2
            else:
                new_obs[j] = obs[i]
                i += 1
            j += 1
        obs = new_obs[:j]
        obs_len = j
        nsyms += 1
    return obs, sym2pair, nsyms


def hmm_forward(
        np.ndarray[double, ndim=1] pi,
        np.ndarray[double, ndim=2] T,
        np.ndarray[double, ndim=2] E,
        np.ndarray[np.int32_t, ndim=1] obs):
    cdef:
        size_t k, L, t
        np.ndarray[double, ndim=2] An
        np.ndarray[double, ndim=1] C, D
        double x, C_n, C_n_inv
        np.int32_t o
    k = T.shape[0]
    L = obs.shape[0]
    An = np.zeros((L, k))
    C = np.zeros(L)
    D = np.zeros(k)
    C_n = 0.0
    for i in xrange(k):
        C_n += pi[i] * E[i, obs[0]]
    C[0] = C_n
    for i in xrange(k):
        An[0, i] = pi[i] * E[i, obs[0]] / C_n
    for t in range(1, L):
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
    return An, C, np.log(C).sum()


def hmm_forward_backward(
        np.ndarray[double, ndim=1] pi,
        np.ndarray[double, ndim=2] T,
        np.ndarray[double, ndim=2] E,
        np.ndarray[np.int32_t, ndim=1] obs):
    cdef:
        size_t k, L, i, j, n
        np.ndarray[double, ndim=2] A, B
        np.ndarray[double, ndim=1] C
        double x
        np.int32_t o
    assert T.shape[0] == T.shape[1]
    k = T.shape[0]
    L = obs.shape[0]
    n = L - 1
    A, C, logL = hmm_forward(pi, T, E, obs)
    B = np.zeros((L, k))
    for i in range(k):
        B[n, i] = 1.0
    while n > 0:
        o = obs[n]
        for i in range(k):
            x = 0.0
            for j in range(k):
                x += B[n, j] * E[j, o] * T[i, j]
            B[n-1, i] = x/C[n]
        n -= 1
    return A, B, C, logL


def hmm_baum_welch(
        np.ndarray[double, ndim=1] pi,
        np.ndarray[double, ndim=2] T,
        np.ndarray[double, ndim=2] E,
        np.ndarray[np.int32_t, ndim=1] obs):
    cdef:
        size_t L, k, i
        np.ndarray[double, ndim=2] A, B, E_counts, T_counts
        np.ndarray[double, ndim=1] C, new_pi
        double logL, tmp, scale
        np.int32_t o
    A, B, C, logL = hmm_forward_backward(pi, T, E, obs)
    T_counts = np.zeros_like(T)
    E_counts = np.zeros_like(E)
    new_pi = np.zeros_like(pi)

    L = len(obs)
    k = T.shape[0]
    o = obs[0]
    for i in range(k):
        tmp = A[0, i]*B[0, i]/C[i]
        new_pi[i] = tmp
        E_counts[i, o] += tmp
    for i in range(1, L):
        scale = 1.0/C[i]
        o = obs[i]
        for j in range(k):
            tmp = A[i-1,j]
            for s in range(k):
                T_counts[j,s] += tmp * T[j, s] * E[s, o] * B[i, s]
            E_counts[j, o] += A[i, j] * B[i, j]
    new_T = T_counts / T_counts.sum(axis=1)[:, np.newaxis]
    new_E = E_counts / E_counts.sum(axis=1)[:, np.newaxis]
    return new_pi, new_T, new_E
