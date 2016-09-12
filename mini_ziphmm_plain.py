from __future__ import division

import numpy as np

_range = range
try: _range = xrange
except: pass


def init_apply_em_prob(pi, E, symbol):
    # states = pi.shape[0]
    # res = np.zeros((states, 1))
    # res[:, 0] = pi * E[:, symbol]
    # return res
    return pi * E[:, symbol]


def apply_em_prob(T, E, symbol):
    states = T.shape[0]
    res = np.zeros((states, states))
    for i in _range(states):
        res[i, :] = T[:, i] * E[i, symbol]
    return res


def make_em_trans_probs_array(T, E):
    sym2matrix = [apply_em_prob(T, E, i) for i in _range(E.shape[1])]
    sym2scale = [mat.sum() for mat in sym2matrix]
    sym2matrix = [mat / scale for mat, scale in zip(sym2matrix, sym2scale)]
    sym2scale = [np.log(x) for x in sym2scale]
    return sym2matrix, sym2scale


def _forward(pi, T, E, obs, sym2mat, sym2scale):
    res = init_apply_em_prob(pi, E, obs[0])
    res_scale = res.sum()
    res /= res_scale
    logL = np.log(res_scale)
    for o in obs[1:]:
        res = np.dot(sym2mat[o], res)
        res_scale = res.sum()
        res /= res_scale
        logL += np.log(res_scale)
        logL += sym2scale[o]
    return logL


def simple_forward(pi, T, E, obs):
    sym2mat, sym2scale = make_em_trans_probs_array(T, E)
    return _forward(pi, T, E, obs, sym2mat, sym2scale)


def forward(pi, T, E, sym2pair, preprocessed_obs, orig_nsyms, nsyms):
    assert len(T.shape) == len(E.shape) == 2
    assert len(pi.shape) == 1
    assert nsyms >= orig_nsyms
    assert E.shape[1] == orig_nsyms
    assert T.shape[0] == T.shape[1] == E.shape[0] == pi.shape[0]
    sym2mat, sym2scale = make_em_trans_probs_array(T, E)
    for i in _range(orig_nsyms, nsyms):
        pair = sym2pair[i]
        left_mat = sym2mat[pair[0]]
        right_mat = sym2mat[pair[1]]
        new_mat = np.dot(right_mat, left_mat) # <- reverse order of seq
        new_mat_scale = new_mat.sum()
        sym2mat.append(new_mat / new_mat_scale)
        sym2scale.append(np.log(new_mat_scale) + sym2scale[pair[0]] + sym2scale[pair[1]])
    return _forward(pi, T, E, preprocessed_obs, sym2mat, sym2scale)


def preprocess_raw_observations(obs, nsyms):
    sym2pair = dict()
    prev_max_count = -1
    while True:
        pair_counts = np.zeros(nsyms**2, dtype=np.int32)
        counted = True
        for idx in _range(1, len(obs)-1):
            p_prev = obs[idx-1] * nsyms + obs[idx]
            p = obs[idx] * nsyms + obs[idx+1]
            if not counted or p_prev != p:
                pair_counts[p] += 1
                counted = True
            else:
                counted = False
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
        # print "=="
        # print "obs:        ", obs
        # print "pair counts:", pair_counts
        # print "max_pair:   ", "index={0} (seq={1}{2})".format(max_pair, max_pair_left, max_pair_right)
        # Replace all occurences of max_pair in obs with new_sym
        new_obs = np.zeros_like(obs)
        i, j = 1, 1
        new_obs[0] = obs[0]
        obs_len = len(obs)
        while i < obs_len:
            if i + 1 < obs_len and max_pair == obs[i]*nsyms + obs[i+1]:
                new_obs[j] = new_sym
                i += 2
            else:
                new_obs[j] = obs[i]
                i += 1
            j += 1
        obs = new_obs[:j]
        nsyms += 1
    # print "=="
    # print "final obs:   ", " ".join(map(str,obs))
    return obs, sym2pair, nsyms
