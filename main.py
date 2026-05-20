# main.py
import numpy as np
import matplotlib.pyplot as plt

from env import PricingBanditEnv
from model import EpsilonGreedy, SlidingWindowUCB
from plot import (
    violinplot_environment, data_average_plot, data_cumulative_plot,
    model_average_plot, model_cumulative_plot
)

def compute_regret(rewards, best_mean):
    T = len(rewards)
    optimal = np.arange(1, T + 1) * best_mean
    return optimal - np.cumsum(rewards)

def create_action_matrix(actions, K):
    matrix = np.zeros((len(actions), K))
    for t, a in enumerate(actions):
        matrix[t, a] = 1
    return matrix

def main():

    # ===== LOAD ENVIRONMENT =====
    env = PricingBanditEnv("online_retail_II.xlsx", K=5)
    data = env.data
    arm_means = [env.arm_means]
    T, K = data.shape

    # ===== ENVIRONMENT PLOTS =====
    violinplot_environment(data)
    data_average_plot(data, arm_means)
    data_cumulative_plot(data, arm_means)

    # ===== BEST ARM =====
    best_arm = np.argmax(env.arm_means)
    best_mean = env.arm_means[best_arm]
    print(f"[INFO] Best arm = {best_arm}, mean reward = {best_mean:.2f}")

    # ===== MULTIPLE RUNS =====
    N_RUNS = 50
    all_rewards_eg = np.zeros((N_RUNS, T))
    all_rewards_sw = np.zeros((N_RUNS, T))

    for r in range(N_RUNS):
        np.random.seed(r)

        eg = EpsilonGreedy(epsilon=0.1, n_arms=K)
        sw = SlidingWindowUCB(n_arms=K, window=30)

        # Run EG
        rewards_eg = np.zeros(T)
        arms_eg = np.zeros(T, dtype=int)
        for t in range(T):
            a = eg.select_arm()
            reward = data[t, a]
            eg.update(a, reward)
            rewards_eg[t] = reward
            arms_eg[t] = a

        # Run SW-UCB
        rewards_sw = np.zeros(T)
        arms_sw = np.zeros(T, dtype=int)
        for t in range(T):
            a = sw.select_arm(t)
            reward = data[t, a]
            sw.update(a, reward)
            rewards_sw[t] = reward
            arms_sw[t] = a

        all_rewards_eg[r, :] = rewards_eg
        all_rewards_sw[r, :] = rewards_sw

    # ===== AVERAGED RESULTS =====
    mean_rewards_eg = all_rewards_eg.mean(axis=0)
    mean_rewards_sw = all_rewards_sw.mean(axis=0)

    # ===== CUMULATIVE PLOTS =====
    plt.figure()
    plt.plot(np.cumsum(mean_rewards_eg), label="Epsilon-Greedy")
    plt.plot(np.cumsum(mean_rewards_sw), label="SW-UCB")
    plt.title("Cumulative Reward Comparison (Averaged Over 50 Runs)")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    plt.show()

    # ===== REGRET PLOTS =====
    mean_regret_eg = compute_regret(mean_rewards_eg, best_mean)
    mean_regret_sw = compute_regret(mean_rewards_sw, best_mean)

    plt.figure()
    plt.plot(mean_regret_eg, label="EG Regret")
    plt.plot(mean_regret_sw, label="SW-UCB Regret")
    plt.title("Regret vs Best Fixed Arm (Averaged Over Runs)")
    plt.xlabel("Time")
    plt.ylabel("Regret")
    plt.legend()
    plt.show()

    # ===== MODEL PLOTS =====
    eg_action_matrix = create_action_matrix(arms_eg, K)
    sw_action_matrix = create_action_matrix(arms_sw, K)

    model_average_plot(data, rewards_eg, eg_action_matrix, arm_means)
    model_cumulative_plot(data, rewards_eg, eg_action_matrix, arm_means)

    model_average_plot(data, rewards_sw, sw_action_matrix, arm_means)
    model_cumulative_plot(data, rewards_sw, sw_action_matrix, arm_means)


if __name__ == "__main__":
    main()
