import argparse
import os
from collections import deque

import numpy as np
import torch
from matplotlib import pyplot as plt
from unityagents import UnityEnvironment

from agents.ddpd_multiagent import MultiAgent
from agents.ddpg_agent import Agent


def ddpg_training(env: UnityEnvironment, n_episodes: int = 2500, max_t: int = 1000, agent_class=Agent):
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)
    print("Number of agents:", num_agents)

    agent = MultiAgent(num_agents, state_size=brain.vector_observation_space_size * brain.num_stacked_vector_observations,
        action_size=brain.vector_action_space_size,
        random_seed=0,)

    scores_deque = deque(maxlen=100)
    scores = []

    solved = False
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
        agent.reset()
        score = np.zeros(len(env_info.agents))
        for _ in range(max_t):
            action = agent.act(state)

            env_info = env.step(action)[brain_name]

            next_state = env_info.vector_observations  # get the next state
            reward = env_info.rewards  # get the reward
            done = env_info.local_done

            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if np.any(done):
                break
        score = np.max(score)
        scores_deque.append(score)
        scores.append(score)
        print(
            "\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}".format(i_episode, np.mean(scores_deque), score), end=""
        )
        if i_episode % 100 == 0:
            print("\rEpisode {}\tAverage Score: {:.2f}".format(i_episode, np.mean(scores_deque)))
        if not solved and np.mean(scores_deque) >= 0.5:
            print(
                "\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}".format(
                    i_episode - 100, np.mean(scores_deque)
                )
            )
            for i, single_agent in enumerate(agent.agents):
                torch.save(single_agent.actor_local.state_dict(), "checkpoint_actor_{i}.pth")
                torch.save(single_agent.critic_local.state_dict(), "checkpoint_critic_{i}.pth")
            solved = True
    return scores


def evaluation(environment_path: str):
    """Evaluates five reinforcement learning models

    Args:
        environment_path: Path to the banana executable

    Returns:

    """
    env = UnityEnvironment(file_name=os.path.join(os.path.dirname(__file__), environment_path), no_graphics=True)

    scores_dqn = ddpg_training(env, agent_class=Agent)

    # plot the scores
    rolling_window = 10
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores_dqn)), scores_dqn, label="ddpg")
    rolling_mean = pd.Series(scores).rolling(rolling_window).mean()
    plt.plot(rolling_mean)
    plt.legend(loc="upper left")
    plt.ylabel("Score")
    plt.xlabel("Episode #")
    plt.savefig("ddpg_scores.pdf")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tennis_executable", help="Path to the tennis environment executable")
    args = parser.parse_args()
    evaluation(args.tennis_executable)
