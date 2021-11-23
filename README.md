# Tennis environment solved with Deep Deterministic Policy Gradients
This repository offers you a DDPG implementation that solves the tennis environment from the Unity ML-Agents toolkit.

The environment consists of two agents that controls two rackets in a tennis court. The rackets can move forward 
(to the net), back and jump. The goal of the environment is to keep the ball being played as long as possible. Thus, 
the agents receive a reward of +0.1 when they throw the ball over the net. If the ball lands out of bounds or hit the 
ground, then the agents receive a reward of -0.01.

The input space consists of 8 float variables that describe the position and velocity of both the agent's racket and 
the ball. The action array contains two elements that describe if the agent has to move the racket forward or away of 
the net and if the racket has to jump to hit the ball. Each agent receives its own local observation, i.e., they are
not aware of the position of the other one. 

When analyzing the environment provided by Udacity colleagues I realized that the agents receive actually a state with
24 variables. This is because they stack three consecutive states together, so we have like a snapshot of the state 
evolution.

The environment is considered to be solved after obtaining an average score greater than 0.5 over 100 episodes. Given 
that we have two agents, we also have two scores. We take the maximum of these two scores after each episode in order 
to compute the average score over 100 episodes. 


# Installation

## 1: Install project dependencies
Please install the pip dependencies with the following command:

<code>pip install -r requirements.txt</code>

## 2: Download the Reacher Unity Environment
Download the version corresponding to your operative system and place the uncompressed content in the root path of this
repository:

### Version 1
* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)


# Running the code
We prepared an evaluation script <code>evaluation.py</code> which trains and produced the weights DDPG model:

You should provide the tennis executable path to the evaluation script like as follows:

<code>python evaluation.py --tennis_executable Tennis_Windows_x86_64/Tennis.exe</code>

Executing the evaluation script will generate the following files in the root folder of this repository:
* A PDF named <code>ddpg_scores.pdf</code> containing a plot of the scores achieved by each model for each timestep.
* Two files called <code>checkpoint_actor_X.pth</code> containing the weights of the actor neural network, where X is 
the agent ID.
* Two files called <code>checkpoint_critic_X.pth</code> containing the weights of the critic neural network, where X is 
the agent ID.
