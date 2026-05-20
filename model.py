# model.py
import numpy as np
import pandas as pd



class EpsilonGreedy:
    """
    Implementation of the Epsilon-Greedy algorithm for Multi-Armed Bandits.

    Parameters:
    - n_arms (int): Number of arms.
    - epsilon (float): Exploration probability.
    - manual_prior (tuple): Optional (Q_init, N_init).
    - random_seed (int): Random seed.

    Methods:
    - bandit(data, A, t): Returns reward for selecting arm A at time t.
    - action(): Selects arm using epsilon-greedy rule.
    - update(A, R): Updates Q and N.
    - train(data): Runs the algorithm over all timesteps.
    - create_table(): Returns N and Q for each arm.
    - save(): Saves Q and N.
    - reset(): Resets learning state.
    """

    def __init__(self, n_arms, epsilon=0.1, manual_prior=None, random_seed=None):
        np.random.seed(random_seed)
        self.n_arms = n_arms
        
        if manual_prior:
            self.Q = np.ones((n_arms, 1)) * manual_prior[0]
            self.N = np.ones((n_arms, 1)) * manual_prior[1]
        else:
            self.Q = np.zeros((n_arms, 1))
            self.N = np.zeros((n_arms, 1))
        
        self.epsilon = epsilon
        self.rewards_list = []
        self.rewards_matrix = np.zeros((1, n_arms))

    def bandit(self, data, A, t):
        reward_vector = data[t].copy()
        R = reward_vector[A]

        reward_vector[:A] = 0
        reward_vector[A + 1:] = 0

        self.rewards_matrix = np.vstack([self.rewards_matrix,
                                         reward_vector.reshape((1, -1))])
        self.rewards_list.append(R)
        return R

    def action(self):
        if np.random.uniform(0, 1) <= self.epsilon:
            return np.random.randint(0, self.n_arms)
        return np.argmax(self.Q)
    
    def select_arm(self, t=None):
        return self.action()


    def update(self, A, R):
        self.N[A] += 1
        self.Q[A] += (1 / self.N[A]) * (R - self.Q[A])

    def train(self, data):
        for t in range(len(data)):
            A = self.action()
            R = self.bandit(data, A, t)
            self.update(A, R)

        table = self.create_table()
        return table, self.rewards_list, self.rewards_matrix[1:, :]

    def create_table(self):
        table = np.hstack([
            np.arange(1, self.n_arms + 1).reshape(self.n_arms, 1),
            self.N,
            self.Q.round(2)
        ])
        df = pd.DataFrame(table, columns=["Arms", "Arm Selection", "E(reward|action)"])
        return df.to_string(index=False)

    def save(self):
        return pd.DataFrame({"N": self.N.flatten(), "Q": self.Q.flatten()})

    def reset(self):
        self.Q[:] = 0
        self.N[:] = 0
        self.rewards_list = []
        self.rewards_matrix = np.zeros((1, self.n_arms))


class SlidingWindowUCB:
    """
    Sliding-Window Upper Confidence Bound (SW-UCB) algorithm.

    Parameters:
    - n_arms (int): Number of arms.
    - window (int): Number of recent rewards to use.
    - random_seed (int): Random seed.

    Methods:
    - bandit(data, A, t): Returns reward for selecting arm A.
    - action(t): Selects arm based on sliding-window UCB rule.
    - update(A, R): Appends reward to arm history.
    - train(data): Runs SW-UCB algorithm.
    - create_table(): Summaries counts and estimates.
    - save(): Saves statistics.
    - reset(): Resets state.
    """

    def __init__(self, n_arms, window=30, random_seed=None):
        np.random.seed(random_seed)
        self.n_arms = n_arms
        self.window = window

        self.rewards = [[] for _ in range(n_arms)]
        self.rewards_list = []
        self.rewards_matrix = np.zeros((1, n_arms))

    def bandit(self, data, A, t):
        reward_vector = data[t].copy()
        R = reward_vector[A]

        reward_vector[:A] = 0
        reward_vector[A + 1:] = 0

        self.rewards_matrix = np.vstack([self.rewards_matrix,
                                         reward_vector.reshape((1, -1))])
        self.rewards_list.append(R)
        return R

    def action(self, t):
        ucbs = []
        for a in range(self.n_arms):
            past = self.rewards[a][-self.window:]

            # unseen arms → infinite UCB
            if len(past) == 0:
                ucbs.append(float("inf"))
            else:
                mean = np.mean(past)
                bonus = np.sqrt(2 * np.log(t + 1) / len(past))
                ucbs.append(mean + bonus)

        return int(np.argmax(ucbs))
    
    def select_arm(self, t):
        return self.action(t)


    def update(self, A, R):
        self.rewards[A].append(R)

    def train(self, data):
        for t in range(len(data)):
            A = self.action(t)
            R = self.bandit(data, A, t)
            self.update(A, R)

        table = self.create_table()
        return table, self.rewards_list, self.rewards_matrix[1:, :]

    def create_table(self):
        means = [np.mean(r[-self.window:]) if len(r) > 0 else 0
                 for r in self.rewards]
        counts = [len(r) for r in self.rewards]

        df = pd.DataFrame({
            "Arms": np.arange(1, self.n_arms + 1),
            "Arm Selection": counts,
            "E(reward|action)": np.round(means, 2)
        })
        return df.to_string(index=False)

    def save(self):
        means = [np.mean(r[-self.window:]) if len(r) > 0 else 0
                 for r in self.rewards]
        counts = [len(r) for r in self.rewards]
        return pd.DataFrame({"N": counts, "Q": means})

    def reset(self):
        self.rewards = [[] for _ in range(self.n_arms)]
        self.rewards_list = []
        self.rewards_matrix = np.zeros((1, self.n_arms))
