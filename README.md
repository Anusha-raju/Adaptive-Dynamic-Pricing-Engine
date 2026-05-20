# RL Project - Pricing Bandit Simulation

## Overview

This repository implements a pricing-oriented multi-armed bandit experiment using real retail sales data. The code loads an online retail dataset, converts daily average price buckets into arms, and evaluates two bandit algorithms:

- **Epsilon-Greedy**
- **Sliding Window Upper Confidence Bound (SW-UCB)**

The main goal is to compare the algorithms on cumulative reward, regret, and action selection behavior.

## Repository Structure

- `main.py` - Entry point for the experiment. Loads environment data, runs multiple simulation runs, and plots results.
- `env.py` - Defines `PricingBanditEnv`, which preprocesses the retail dataset and generates the bandit reward matrix.
- `model.py` - Implements the `EpsilonGreedy` and `SlidingWindowUCB` algorithms.
- `plot.py` - Contains visualization routines for environment distributions and algorithm performance.
- `results/` - Intended folder for storing experiment outputs or exported files.
- `README.md` - Project documentation.

## Requirements

The project requires Python 3.8+ and the following packages:

- `numpy`
- `pandas`
- `matplotlib`
- `statsmodels`

If you use a virtual environment, activate it before installing dependencies.

### Install dependencies

```powershell
python -m pip install numpy pandas matplotlib statsmodels
```

## Dataset

The project expects the file `online_retail_II.xlsx` to be available in the same directory as `main.py`.

This workbook is loaded by `PricingBanditEnv` and concatenated across all sheets. The environment filters sales data, selects the top country and top-selling SKU, then divides price values into `K` buckets to define bandit arms.

> If you do not have the dataset, place it in the project folder or update `env.py` to point to your own Excel file.

## How It Works

### Environment (`PricingBanditEnv`)

- Loads the Excel workbook and concatenates sheets.
- Filters out invalid records with non-positive quantities or prices.
- Converts invoice timestamps to dates and computes daily revenue.
- Selects the country and SKU with the highest sales volume.
- Aggregates daily revenue and average price.
- Assigns each day to one of `K` price buckets.
- Builds a reward matrix of shape `(T, K)` where each row is a timestep and each column is an arm.
- Computes the true mean revenue for each arm and prints stationarity report results.

### Algorithms

#### Epsilon-Greedy

- Picks a random arm with probability `epsilon`.
- Otherwise selects the arm with the highest estimated mean reward.
- Updates sample counts and reward estimates after each timestep.

#### Sliding Window UCB

- Uses the most recent `window` rewards for each arm.
- Computes an upper confidence bound per arm.
- Selects the arm with the highest UCB value.
- Adapted for non-stationary reward streams by limiting history to the sliding window.

## Running the Experiment

Run the experiment from the project root:

```powershell
python main.py
```

The script performs the following steps:

1. Loads and preprocesses the dataset using `PricingBanditEnv`.
2. Displays environment-level plots:
   - Violin plot of reward distribution per arm.
   - Rolling average reward per arm.
   - Cumulative reward per arm.
3. Identifies the best arm using true arm means.
4. Runs `N_RUNS = 50` independent trials of both algorithms.
5. Plots average cumulative reward and regret curves.
6. Generates model-specific action/reward plots for each algorithm.

## Output

The script displays plots using `matplotlib`. Key outputs include:

- Environment distribution plots
- Average rewards over time
- Cumulative reward comparison
- Regret comparison vs best fixed arm
- Model action and reward visualizations

## Customization

You can modify the experiment behavior in `main.py`:

- `K` in `PricingBanditEnv` to choose the number of arms
- `N_RUNS` to increase or reduce simulation repetitions
- `epsilon` for the Epsilon-Greedy algorithm
- `window` for the Sliding Window UCB algorithm

## Notes

- `main.py` currently uses the final trial rewards from the last simulation run to generate model plots.
- `PricingBanditEnv` uses quantile-based bucketization, with fallback to equal-width buckets if necessary.
- Stationarity is tested using the Augmented Dickey-Fuller statistic for each arm.

## Suggestions for Extension

- Add a new bandit algorithm such as Thompson Sampling or UCB1.
- Save experiment results into CSV files inside `results/`.
- Add command-line arguments for dataset path, algorithms, and experiment parameters.
- Implement a train/test split or time-varying reward structure.

## Contact

If you want to extend this experiment or adapt it to your own pricing data, update `env.py` to load your dataset and tune `main.py` accordingly.
