from importlib.resources import path
import pandas as pd
import numpy as np
from scipy.stats import norm


# The Geometric Brownian motion price simulation is able to generate the price at time T directly
def gbm_price(S0, r, sigma, T, rng):
    error = rng.standard_normal()  # generate standard normal random variables
    S = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * error)
    return S


# The Euler discretization simulates the price path step by step
def euler_discretization(S0, r, sigma, T, dt, rng):
    N = int(T/dt)  # number of time steps
    times = np.linspace(0, T, N+1)
    S = {times[0]: S0}
    
    for t in range(1, N+1):
        error = rng.standard_normal()  # generate standard normal random variables
        S[times[t]] = S[times[t-1]] *(1 + r * dt + sigma * np.sqrt(dt) * error)
    return S


# Black-Scholes formula for European call option price
def black_scholes_call(S0, K, r, T, sigma):
    d1= (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2= d1 - sigma * np.sqrt(T)
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    call_price = S0 * Nd1 - K * np.exp(-r * T) * Nd2
    return call_price


# Monte Carlo simulation using Euler discretization
def MonteCarlo_sim_euler(S0, r, sigma, T, dt, n_simulations, rng):
    simulations = []
    for _ in range(n_simulations):
        sim = euler_discretization(S0, r, sigma, T, dt, rng)
        simulations.append(sim)
    return simulations


# Graphing price paths for the simulations
def graph_simulations(simulations):
    import matplotlib.pyplot as plt

    for sim in simulations:
        times = list(sim.keys())
        prices = list(sim.values())
        plt.plot(times, prices)

    plt.title('Monte Carlo Simulations of Stock Price Paths')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.show()

if __name__ == "__main__":

    seed = 123
    rng = np.random.default_rng(seed)

    S0 = 100  # initial stock price
    r = 0.05  # risk-free rate
    sigma = 0.2  # volatility
    T = 1.0  # time to maturity in years
    dt = 0.01  # time step
    num_simulations = 100  # number of Monte Carlo simulations



    