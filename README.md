# Mini Project for CS4246: AI Planning and Decision Making
A Deep Q-Learning (DQN) agent in a large stochastic environment with reward shaping and curriculum learning to solve the problem of sparse reward and slow learning

## Authors
Team Member 1: Solayappan Meenakshi
Team Member 2: Liu Zechu

## Project Instructions
Please refer to [`MiniProject_Instructions.pdf`](https://github.com/LiuZechu/CS4246-mini-project/blob/main/MiniProject_Instructions.pdf) for detailed instructions.

### Summary
In this task, the agent must be able to solve a large stochastic environment with the dimension 50 × 10, which is 10 lanes with each having a width of 50. Each lane will have 6-8 cars with speed ranging from 1 to 3 cells per time step. The environment is stochastic, which means that other cars will have a non-deterministic speed. At every time step, the car will move at a different speed, uniformly sampled within the range of minimum and maximum speed of the lane. 

Due to its stochastic nature, the agent will be evaluated on 600 **different** environment configurations of the same size and details. Your agent should be able to run on all test cases within 600 seconds. Half the test cases are restricted to a horizon of 40 (must reach the goal in 40 steps to be successful) while the remaining half have a horizon of 50.

**The goal** is to build an agent that meet the following criterion:
- Agent should be be able to handle stochasticity.
- Agent can only have access to an observation, not a state, in the form of image (tensor).
- In the testing server, agent will have no access to environment’s MDP, or its simulator. During training, you can access everything.
- Agent should be optimized to get to the goal in less than 40 steps.
- Its runtime should be reasonably fast since it’s going to be run on 600 different environments within set time limit (600 seconds).

## Project Deliverables
Please refer to [`write_up.pdf`](https://github.com/LiuZechu/CS4246-mini-project/blob/main/write_up.pdf) for a write-up on how we designed our agent.

Please refer to the [`source_code`](https://github.com/LiuZechu/CS4246-mini-project/tree/main/source_code) directory for our code submission. Note that the code has to be run with the docker image provided in the instructions PDF file.

### Notable files inside `source_code` directory
- `agent/init.py`: main entry point of the agent 
- `env.py`: definition of environment
- `models.py`: neural networks for function approximations
- `env_easy.py`: definition of an "easier" environment for the model to train on initially
- `dqn_train.py`: **the main bulk of the work is here**. This is where Deep Q-Learning happens. This file trains and outputs progressively better models saved as `.pt` files. Gradually increment `TRAINING_SESSION_NUMBER` in Line 27 from 1 to 6 and run this file with `python --train dqn_train.py` to train and save the models.
