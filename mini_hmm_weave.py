from scipy import zeros, array
import scipy.weave as weave
import numpy as np


def calc_forward(pi, T, E, obs):
    k = len(T)
    L = len(obs)
    An = zeros((L, k), dtype=np.float64)
    C = zeros(L, dtype=np.float64)
    D = zeros(k, dtype=np.float64)

    code = """
    size_t t,j,i,o;
    double x, C_n;
    C_n = 0.0;
    for (i = 0; i < k; i++)
        C_n += pi[i] * E2(i, obs[0]);
    C[0] = C_n;
    for (i = 0; i < k; i++)
        An[0*k + i] = pi[i] * E2(i, obs[0]) / C_n;
    for (t = 1; t < L; t++)
    {
        o = obs[t];
        for (j = 0; j < k; j++)
        {
            x = 0;
            for (i = 0; i < k; i++)
                x += T2(i, j) * AN2(t-1, i);
            D[j] = x * E2(j, o);
        }
        C_n = 0.0;
        for (j = 0; j < k; j++)
            C_n += D[j];
        C[t] = C_n;
        for (j = 0; j < k; j++)
            AN2(t, j) = D[j]/C_n;
    }
    x = 0.0;
    for (t = 0; t < L; t++)
        x += log(C[t]);
    return_val = x;
    """
    res = weave.inline(code,
                       ['k', 'L', 'An', 'C', 'D', 'pi', 'T', 'E', 'obs'],
                       compiler="gcc")
    return An, C, res


def calc_forward_backward(pi, T, E, obs):
    A, C, logL = calc_forward(pi, T, E, obs)
    k = len(T)
    L = len(obs)
    Ew = E.shape[1]
    B = zeros((L, k), dtype=np.float64)
    B[L-1, :] = 1.0
    code = """
    size_t n, i, j;
    double x;
    int symb;
    for (n = L - 1; n > 0; n--)
    {
        symb = obs[n];
        for (i = 0; i < k; i++)
        {
            x = 0.0;
            for (j = 0; j < k; j++)
            {
                x += B2(n,j)*E2(j,symb)*T2(i,j);
            }
            B2(n-1,i) = x/C[n];
        }
    }
    """
    weave.inline(code,
                 ['k', 'L', 'C', 'B', 'T', 'E', 'obs', 'Ew'],
                 compiler="gcc")
    return A, B, C, logL


def baum_welch(pi, T, E, obs):
    L = len(obs)
    k = len(T)
    A, B, C, logL = calc_forward_backward(pi, T, E, obs)
    pi_counts = zeros(pi.shape)
    T_counts = zeros(T.shape)
    E_counts = zeros(E.shape)
    new_pi = zeros(pi.shape)

    x = obs[0]
    for i in range(k):
        tmp = A[0, i]*B[0, i]/C[i]
        new_pi[i] = tmp
        E_counts[i, x] += tmp
    code = """
    int i,j,s,x;
    for (i = 1; i < L; i++)
    {
        double scale = 1.0/C[i];
        x = obs[i];
        for (j = 0; j < k; j++)
        {
            double tmp = A2(i-1,j);
            for (s = 0; s < k; s++)
            {
                T_COUNTS2(j,s) += tmp * T2(j,s) * E2(s,x) * B2(i,s);
            }
            E_COUNTS2(j,x) += A2(i,j) * B2(i,j);
        }
    }
    """
    weave.inline(code,
                 ['k', 'L', 'A', 'B', 'C',
                     'T', 'E', 'obs', 'E_counts', 'T_counts'],
                 compiler="gcc")
    new_T = T_counts / T_counts.sum(axis=1)[:, np.newaxis]
    new_E = E_counts / E_counts.sum(axis=1)[:, np.newaxis]
    assert all(abs(row.sum() - 1.0) < 0.001 for row in new_T)
    assert all(abs(row.sum() - 1.0) < 0.001 for row in new_E)
    return new_pi, new_T, new_E
