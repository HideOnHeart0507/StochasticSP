import numpy as np
from scipy.stats import binom, poisson
import time

"""  Ths script acompined the paper "The service points location and capacity problem" by Tal Raviv (2022)

    It has two purposes 
    1) Run the experiment reported in Section 3.3 
    2) To be used as package with the functions "markov_chain_sp(C, lamb, p)" documented below
        "sim_sp(C, lamb, pd, N=1000000, Warmup=100, rand_seed=0)"

    Copyright Tal Raviv (2022). Feel free to use it but please let my know if you do  (talraviv@tau.ac.il)
    Last updated: June 22, 2022
"""

"""  markov_chain_sp_new(C_s, C_L, lamb_s, lamb_L, p)
        C_s - Station capacity for small parcel lockers
        C_L - Station capacity for small parcel lockers
        lamb_s - supply rate for small parcels
        lamb_L - supply rate for large parcels
        p - pickup probability 

    The function returns 
        R - the mean number of rejections per period  
        CPU_time - the computation time
"""


def markov_chain_sp(C_s, C_L, lamb_s, lamb_L, p_s, p_L):
    my_inf = int(
        (max(C_s, C_L) + max(lamb_s, lamb_L)) * 2)  # large enough to ignore the tail of the Poisson distribution

    start_time = time.time()
    P = np.zeros(((C_s + C_L + 2) * 2, (C_s + C_L + 2) * 2))  # Initialized transition probability matrix

    """ Populate the transition probability matrix """
    # Iterate over all possible states
    for i_s in range(C_s + 1):
        for i_L in range(C_L + 1):
            for j_s in range(i_s + 1):
                for j_L in range(i_L + 1):
                    # State index calculation
                    index_from = i_s * (C_L + 1) + i_L
                    index_to = (C_s + 1 + j_s) * (C_L + 1) + j_L

                    # Transition probabilities for various conditions
                    if i_s == C_s and i_L < C_L:
                        P[index_from, index_to] = poisson.cdf(C_s - j_s, lamb_s) * poisson.pmf(i_L - j_L, lamb_L)
                    elif i_L == C_L and i_s < C_s:
                        P[index_from, index_to] = poisson.pmf(i_s - j_s, lamb_s) * poisson.cdf(C_L - j_L, lamb_L)
                    elif i_s == C_s and i_L == C_L:
                        P[index_from, index_to] = poisson.cdf(C_s - j_s, lamb_s) * poisson.cdf(C_L - j_L, lamb_L)
                    else:
                        P[index_from, index_to] = poisson.pmf(i_s - j_s, lamb_s) * poisson.pmf(i_L - j_L, lamb_L)

    # Steady state probabilities
    ssp = np.linalg.matrix_power(P, 512)[-1, (C_s + 1) * (C_L + 1):]

    """ Calculate R """
    R = 0
    for j_s in range(C_s + 1):
        for j_L in range(C_L + 1):
            for k in range(C_s - j_s + 1, my_inf):
                for l in range(C_L - j_L + 1, my_inf):
                    ssp_index = j_s * (C_L + 1) + j_L
                    R += ssp[ssp_index] * poisson.pmf(k, lamb_s) * poisson.pmf(l, lamb_L) * (k + l - C_s - C_L)

    return R, time.time() - start_time


"""   Simulation model with the emperical time-to-pickup distribution

      sim_sp(C, lamb, pd)
        C - Station capacity
        lamb - supply rate
        pd - time-to-pickup emperical distribution 
        N - number of periods in the simulation (default 1,000,000)
        Warmup - number of warmup periods (default 100)
        rand_seed - (default 0)

    The function returns 
        avg - the estimated mean number of rejections per period  
        hci99, hci95 - half of the width of the 99 and 95 confidence intervals 
        CPU_time - the computation time
"""


def sim_sp(C, lamb, pd, N=1000000, Warmup=100, rand_seed=0):
    np.random.seed(rand_seed)
    start_time = time.time()
    H = len(pd)
    departures = np.zeros(N + H)
    R = np.zeros(N)
    I = 0

    for t in range(N):
        NewArrivals = np.random.poisson(lamb)
        rt = np.random.choice(range(H), NewArrivals, p=pd)
        for i in rt:
            if I < C:
                departures[t + i] += 1
                I += 1
            else:
                R[t] += 1
        I -= departures[t]

    """  report sumulation statistics """
    CPU_time = time.time() - start_time
    avg = np.average(R[Warmup:])
    std = np.std(R[Warmup:])
    return avg, std, CPU_time


"""  Run the small expierment of Section 3.3 in the paper - June 2022 version"""

if __name__ == '__main__':
    Z99 = 2.576  # Z values for confidence interval (assuming N is large)
    Z95 = 1.96
    for pd in [np.array([0.5, 0.2, 0.3]),
               np.array([0.4, 0.2, 0.1, 0.08, 0.05, 0.02, 0.15])]:  # time-to-pickup distribution
        H = len(pd)  # max pickup  periods
        p = 1 / np.dot(pd, np.array(range(1, H + 1)))  # pickup probability  (geometric time-to-pickup)
        for C in [20, 50, 100]:  # SP capacity
            for rho in [0.9, 0.95, 0.99, 1, 1.01, 1.05, 1.1]:
                lamb = rho * C * p  # supply rate
                # print(f"Running experiment with H={H}, p={p:.4f}, C={C}, rho={rho}, lambda={lamb:.4f}")
                R_markov, cpu_time_markov = markov_chain_sp(C, lamb, p)
                avg, std, cpu_time_sim = sim_sp(C, lamb, pd)
                hci99 = Z99 * std / np.sqrt(999900)
                hci95 = Z95 * std / np.sqrt(999900)
                print(
                    f"{H}, {p}, {C}, {rho}, {lamb}, {R_markov}, {cpu_time_markov}, {avg}, {std}, {hci99}, {hci95}, {cpu_time_sim}")
