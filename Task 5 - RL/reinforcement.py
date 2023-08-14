# methods defined for Task5 - Reinforcement Learning

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from collections import defaultdict

import gymnasium as gym

#################################################################################################################################


def plot_learning_rewards_epsilon(x, scores, epsilons, filename, lines=None):
    """Creates a plot of the training performance."""
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Game", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    # ax2.xaxis.tick_top()
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    # ax2.set_xlabel('x label 2', color="C1")
    ax2.set_ylabel('Score', color="C1")
    # ax2.xaxis.set_label_position('top')
    ax2.yaxis.set_label_position('right')
    # ax2.tick_params(axis='x', colors="C1")
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(fname=filename)

#################################################################################################################################

# custom environment for our smart charging task


HOURS = 2
NUMBER_OF_TIMESTEPS = 8
MEAN = 32
STANDARD_DEVIATION = 6
BATTERY_CAPACITY = 44


class SmartChargingEnv(gym.Env):

    def __init__(self):
        # The number of timesteps (e.g. 2pm-4pm := 8 timesteps)
        self.time_limit = NUMBER_OF_TIMESTEPS
        self.battery_capacity = BATTERY_CAPACITY  # The battery capacity
        self.time_coefficient = HOURS / NUMBER_OF_TIMESTEPS
        self.penalty_out_of_energy = -10000
        self.max_charging_rate = 24

        self.observation_space = gym.spaces.Tuple([gym.spaces.Discrete(
            self.time_limit + 1, start=0), gym.spaces.Discrete(self.battery_capacity + 1, start=0)])

        # We have 4 actions, corresponding to "zero", "low", "medium", "high"
        self.action_space = gym.spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the charging values.
        """
        charging_rates = np.linspace(0, self.max_charging_rate, 4)
        self._action_to_charging_rates = {
            0: charging_rates[0],
            1: charging_rates[1],
            2: charging_rates[2],
            3: charging_rates[3],
        }

    def _get_obs(self):
        return (self._current_time, self._agent_battery_level)

    def _get_info(self):
        return {
            "remaining_capacity": self.battery_capacity - self._agent_battery_level
        }

    def reset(self, seed=None, options=None):
        self._current_time = self.time_limit

        self._agent_battery_level = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        # Map the action to the charging rate in kWh
        charging_rate = self._action_to_charging_rates[action] * \
            self.time_coefficient

        # Update baterry level and time
        self._agent_battery_level = np.clip(
            self._agent_battery_level + charging_rate, 0, self.battery_capacity
        )
        self._current_time -= 1
        # An episode is done if the time limit is reached
        terminated = (0 == self._current_time)

        # calculate reward
        reward = 0
        if terminated:
            energy_demand = np.random.normal(MEAN, STANDARD_DEVIATION)
            reward = (self._agent_battery_level < energy_demand) * \
                self.penalty_out_of_energy
        reward -= self._get_charging_cost(charging_rate)

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info

    def _get_charging_cost(self, charging_rate):
        return math.exp(charging_rate)

    def render(self):
        pass

    def close(self):
        pass

#################################################################################################################################


def create_grids(q_values):
    """Create value and policy grid given q_values from an agent."""
    # convert our state-action values to state values
    # and build a policy dictionary that maps observations to actions
    state_value = defaultdict(float)
    policy = defaultdict(lambda: -1)
    for observation, action_values in q_values.items():
        state_value[observation] = float(np.max(action_values))
        policy[observation] = int(np.argmax(action_values))

    timstep, battery_level = np.meshgrid(
        np.arange(1, 9),
        np.arange(0, BATTERY_CAPACITY + 1, 2),
    )

    # create the value grid for plotting
    value = np.apply_along_axis(
        lambda obs: state_value[(obs[0], obs[1])],
        axis=2,
        arr=np.dstack([timstep, battery_level]),
    )
    value_grid = battery_level, timstep, value

    timstep, battery_level = np.meshgrid(
        np.arange(8, 0, -1),
        np.arange(0, BATTERY_CAPACITY + 1, 2),
    )

    # create the policy grid for plotting
    policy_grid = np.apply_along_axis(
        lambda obs: policy[(obs[0], obs[1])],
        axis=2,
        arr=np.dstack([timstep, battery_level]),
    )
    return value_grid, policy_grid


def create_policy_plots(value_grid, policy_grid):
    """Creates a plot using a value and policy grid."""
    # create a new figure with 2 subplots (left: state values, right: policy)
    timstep, battery_level, value = value_grid
    fig = plt.figure(figsize=plt.figaspect(0.4))
    # fig.suptitle(fontsize=16)

    # plot the state values
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(
        timstep,
        battery_level,
        value,
        rstride=1,
        cstride=1,
        cmap="viridis",
        edgecolor="none",
    )
    plt.xticks(range(0, BATTERY_CAPACITY + 1, 6),
               range(0, BATTERY_CAPACITY + 1, 6))
    plt.yticks(range(1, 9), range(1, 9))
    ax1.set_title(f"State values: ")
    ax1.set_xlabel("Battery Level")
    ax1.set_ylabel("Time left")
    ax1.zaxis.set_rotate_label(False)
    ax1.set_zlabel("Value", fontsize=14, rotation=90)
    ax1.view_init(20, 220)

    # plot the policy
    fig.add_subplot(1, 2, 2)
    cmap = ["grey", "red", "orange", "yellow", "green"]
    ax2 = sns.heatmap(policy_grid, linewidth=0,
                      annot=True, cmap=cmap, cbar=False)
    ax2.set_title(f"Policy: ")
    ax2.set_xlabel("Time left")
    ax2.set_ylabel("Battery Level")
    ax2.set_xticklabels(range(8, 0, -1))
    ax2.set_yticklabels(
        range(0, BATTERY_CAPACITY + 1, 2))

    # add a legend
    legend_elements = [
        Patch(facecolor="grey", edgecolor="black", label="None"),
        Patch(facecolor="red", edgecolor="black", label="0"),
        Patch(facecolor="orange", edgecolor="black", label="8"),
        Patch(facecolor="yellow", edgecolor="black", label="16"),
        Patch(facecolor="green", edgecolor="black", label="24"),
    ]
    ax2.legend(handles=legend_elements, bbox_to_anchor=(1.3, 1))
    plt.show()

    return fig

#################################################################################################################################


def plot_training_performance(env, agent):
    """Creates a plot of the training performance."""
    rolling_length = 500
    fig, axs = plt.subplots(ncols=2, figsize=(16, 5))
    axs[0].set_title("Episode rewards")
    # compute and assign a rolling average of the data to provide a smoother graph
    reward_moving_average = (
        np.convolve(
            np.array(env.return_queue).flatten(), np.ones(rolling_length), mode="valid"
        )
        / rolling_length
    )
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
    # axs[0].set_yscale("symlog")
    axs[1].set_title("Training Error")
    training_error_moving_average = (
        np.convolve(np.array(agent.training_error),
                    np.ones(rolling_length), mode="same")
        / rolling_length
    )
    axs[1].plot(range(len(training_error_moving_average)),
                training_error_moving_average)
    # axs[1].set_yscale("symlog")
    plt.tight_layout()
    plt.show()

#################################################################################################################################


def run_model(env, agent):
    """method to run one episode"""
    agent_epsilon = agent.epsilon
    agent.epsilon = 0
    terminated = False

    score = 0
    actions = []
    battery_levels = [0]

    observation = env.reset()[0]
    while not terminated:
        action = agent.get_action(observation, env)
        observation, reward, terminated, truncated, info = env.step(
            action)
        score += reward

        actions.append(action)
        battery_levels.append(observation[1])

    agent.epsilon = agent.epsilon

    return actions, battery_levels, score


def plot_model_run(battery_levels):
    plt.plot(battery_levels)
    plt.title("Charging process")
    plt.ylabel("Battery Level")
    plt.xlabel("Time")
    # plt.annotate('%d' % battery_levels[-1],
    #             xy=(8, battery_levels[-1]), xytext=(8,  battery_levels[-1]))
    plt.show()
