import numpy as np
from statsmodels.sandbox.stats.runs import runstest_1samp
from statsmodels.tsa.stattools import acf

def entropy(seq, max_val=1024):
    '''
    Calculate the entropy of the given sequence
    '''
    counts = np.bincount(seq - 1, minlength=max_val)
    p = counts / len(seq)
    p = p[p > 0]
    return -np.sum(p * np.log2(p)) / np.log2(max_val)

def runsTest(seq):
    z_stat, p_value = runstest_1samp(seq, cutoff='median', correction=True)
    return int(p_value > 0.05)


def piTest(seq, max_val=1024):
    seq_u = seq / max_val
    length = len(seq) // 2
    u_1 = seq_u[:length]
    u_2 = seq_u[length:]
    norm = np.sqrt(u_1**2 + u_2**2)
    pi_est = 4 * np.mean(norm <= 1)
    return np.exp(-(pi_est-np.pi)**2 / 2)

def acfTest(seq, nlags=200):
    acf_values = np.abs(acf(seq, nlags=nlags, fft=True)) # with fft
    acf_mean = np.mean(acf_values)

    return max(0.0, 1 - acf_mean)

def uniformityTest(seq, max_val=1024):
    counts = np.bincount(seq - 1, minlength=max_val)
    p = counts / len(seq)
    D = np.sum(np.abs(p - 1/max_val)) / 2
    return max(0.0, 1.0 - D)

def RepetitionScore(seq, min_overlap=100, max_val=1024):
    n = len(seq)
    counts = np.bincount(seq - 1, minlength=max_val)
    chance = np.sum((counts / n) ** 2)
    strength = 0.0

    for k in range(1, n - min_overlap + 1):
        match_rate = np.mean(seq[:-k] == seq[k:])
        s = (match_rate - chance) / (1 - chance)
        if s > strength:
            strength = s

    return max(0.0, 1.0 - strength)
