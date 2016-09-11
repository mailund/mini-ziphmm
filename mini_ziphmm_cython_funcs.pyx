# cython: boundscheck=False, wraparound=False
from __future__ import division

import numpy as np

cimport cython
cimport numpy as np
cimport libc.math


def _forward(
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
        for j in range(N):
            res[j] = tmp[j] / res_scale
        logL += libc.math.log(res_scale)
        logL += sym2scale[o]
    return logL


def preprocess_raw_observations(
        np.ndarray[np.int32_t, ndim=1] obs,
        np.int nsyms):
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
