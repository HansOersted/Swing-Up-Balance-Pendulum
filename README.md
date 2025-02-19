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
theta is the pendulum's deviation from the upright position.  
dtheta is the angular velocity.  
u is the control torque applied.

## 3. DDPG

## 4. Remarks
