# The Swing Up Balance Pendulum from MathWorks

This repository contains a reinforcement learning (RL) implementation for the Swing Up Balance Pendulum, based on MathWorks' SimplePendulumWithImage-Continuous environment. The objective is to train an RL agent to swing the pendulum up and balance it using Deep Deterministic Policy Gradient (DDPG).

## 1. The goal and the inputs

The goal is to train an RL agent that can swing up the pendulum from a downward position and maintain balance at the upright position by applying continuous torque.

The inputs are the image of the pendulum, indicating the state, and the angular velocity.
Note that this setting is to align the real case in the environment, where the state, e.g., the angle, is sometimes not achievable directly.

## 2. The rewards

The reward function is designed to encourage stability in the upright position while minimizing oscillations and excessive control effort:

![image](https://github.com/user-attachments/assets/d71090a3-f0f9-4e2b-8f24-9ee150a11d00)


Notations:  
$\theta$ is the pendulum's deviation from the upright position.  
$\dot{\theta}$ is the angular velocity.  
$u$ is the control torque applied.

## 3. Deep Deterministic Policy Gradient (DDPG)

DDPG is used to train the agent, which consists of:

- **Actor Network**: A neural network that takes in observations and outputs continuous actions.
- **Critic Network**: A neural network that estimates the Q-value of a given `(state, action)` pair.

***The entire updating mechanism is summarized below:***  
*Step 1.* Take an action based on the Actor.  
*Step 2.* Record the state and update the reward and Q value.  
*Step 3.* Update the Actor Network.  
*Step 4.* Update the Critic Network.
*Step 5.* The training ends after the reward within an episode reaches a user-set value.

![image](https://github.com/user-attachments/assets/240a6a61-0ccf-4da2-a8f2-ee8af5a62d4e)


## 4. Remarks

Since the final goal is to stabilize the pendulum, the bonus for being near the target state can be added in the reward to enhance the desired final state.  
For example, add a bonus value when $|\theta| \leq 0.1$ and $|\dot{\theta}| \leq 0.1$

