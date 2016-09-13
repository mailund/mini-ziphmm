from __future__ import division

import numpy as np
from . import cython_funcs as c


preprocess_raw_observations = c.preprocess_raw_observations
hmm_baum_welch = c.hmm_baum_welch
hmm_forward_backward = c.hmm_forward_backward
hmm_viterbi = c.hmm_viterbi


def _apply_em_prob(T, E, symbol):
    states = T.shape[0]
    res = np.zeros((states, states))
    for i in range(states):
        res[i, :] = T[:, i] * E[i, symbol]
    return res


def _make_em_trans_probs_array(T, E):
    sym2matrix = [_apply_em_prob(T, E, i) for i in range(E.shape[1])]
    sym2scale = [mat.sum() for mat in sym2matrix]
    sym2matrix = [mat / scale for mat, scale in zip(sym2matrix, sym2scale)]
    sym2scale = [np.log(x) for x in sym2scale]
    return sym2matrix, sym2scale


def zip_forward(pi, T, E, sym2pair, preprocessed_obs, orig_nsyms, nsyms):
    assert len(T.shape) == len(E.shape) == 2
    assert len(pi.shape) == 1
    assert nsyms >= orig_nsyms
    assert E.shape[1] == orig_nsyms
    assert T.shape[0] == T.shape[1] == E.shape[0] == pi.shape[0]
    sym2mat, sym2scale = _make_em_trans_probs_array(T, E)
    for i in range(orig_nsyms, nsyms):
        pair = sym2pair[i]
        left_mat = sym2mat[pair[0]]
        right_mat = sym2mat[pair[1]]
        new_mat = np.dot(right_mat, left_mat)  # <- reverse order of seq
        new_mat_scale = new_mat.sum()
        sym2mat.append(new_mat / new_mat_scale)
        scale = np.log(new_mat_scale) + sym2scale[pair[0]] + sym2scale[pair[1]]
        sym2scale.append(scale)
    sym2mat = np.array(sym2mat)
    sym2scale = np.array(sym2scale)
    return c.zip_forward(pi, T, E, preprocessed_obs, sym2mat, sym2scale)


def hmm_forward(pi, T, E, obs):
    A, C, logL = c.hmm_forward(pi, T, E, obs)
    return logL


def hmm_posterior_decoding(pi, T, E, obs):
    A, B, _, _ = hmm_forward_backward(pi, T, E, obs)
    return A * B
