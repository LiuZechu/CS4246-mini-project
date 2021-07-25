# Group 11
# Team Member 1: Solayappan Meenakshi A0172687N
# Team Member 2: Liu Zechu A0188295L
# Collaborators: None
# Sources: None

import gym
import gym_grid_driving
import collections
import numpy as np
import random
import math
import os

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from gym_grid_driving.envs.grid_driving import LaneSpec
from models import AtariDQN
from env import construct_task_env
from env_easy import construct_easy_task_env

# NOTE: Gradually increment this session number from 1 to 6 for Curriculum Learning
TRAINING_SESSION_NUMBER = 6 # Ranges from 1 to 6 inclusive

# Maps training session number to training params
TRAINING_PARAMS = {
    1: {
        "with_reward_shaping": True,
        "is_easy_env": True,
        "num_cars_per_lane": 1,
        "randomise_env": False,
        "previous_model_path": "model.pt", # dummy value since session 1 trains from scratch
        "new_model_path": "model_1.pt",
        "max_epsilon": 1.0,
        "num_episodes": 2000
    },
    2: {
        "with_reward_shaping": True,
        "is_easy_env": True,
        "num_cars_per_lane": 2,
        "randomise_env": False,
        "previous_model_path": "model_1.pt", 
        "new_model_path": "model_2.pt",
        "max_epsilon": 1.0,
        "num_episodes": 2000
    },
    3: {
        "with_reward_shaping": False,
        "is_easy_env": True,
        "num_cars_per_lane": 4,
        "randomise_env": False,
        "previous_model_path": "model_2.pt", 
        "new_model_path": "model_3.pt",
        "max_epsilon": 1.0,
        "num_episodes": 2000
    },
    4: {
        "with_reward_shaping": False,
        "is_easy_env": True,
        "num_cars_per_lane": 6,
        "randomise_env": False,
        "previous_model_path": "model_3.pt",
        "new_model_path": "model_4.pt",
        "max_epsilon": 1.0,
        "num_episodes": 2000
    },
    5: {
        "with_reward_shaping": False,
        "is_easy_env": False,
        "num_cars_per_lane": None, # not needed since we train on full environment
        "randomise_env": True,
        "previous_model_path": "model_4.pt",
        "new_model_path": "model_final.pt",
        "max_epsilon": 1.0,
        "num_episodes": 6000
    },
    6: {
        "with_reward_shaping": False,
        "is_easy_env": False,
        "num_cars_per_lane": None, # not needed since we train on full environment
        "randomise_env": True,
        "previous_model_path": "model_final.pt",
        "new_model_path": "model_final_improved.pt",
        "max_epsilon": 0.01,
        "num_episodes": 10000
    }
}

# NOTE: Modify training parameters below before each training session for Curriculum Learning.
# Training parameters:
with_reward_shaping = TRAINING_PARAMS[TRAINING_SESSION_NUMBER]["with_reward_shaping"]
is_easy_env = TRAINING_PARAMS[TRAINING_SESSION_NUMBER]["is_easy_env"]
num_cars_per_lane = TRAINING_PARAMS[TRAINING_SESSION_NUMBER]["num_cars_per_lane"]
randomise_env = TRAINING_PARAMS[TRAINING_SESSION_NUMBER]["randomise_env"]

# Setting up
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
script_path = os.path.dirname(os.path.realpath(__file__))
previous_model_path = os.path.join(script_path, 'model_final.pt')
new_model_path = os.path.join(script_path, 'model_final_improved.pt')

# Hyperparameters --- don't change, RL is very sensitive
learning_rate = 0.001
gamma         = 0.98
buffer_limit  = 5000
batch_size    = 32
max_episodes  = TRAINING_PARAMS[TRAINING_SESSION_NUMBER]["num_episodes"]
t_max         = 40
min_buffer    = 1000
target_update = 20 # episode(s)
train_steps   = 10
max_epsilon   = TRAINING_PARAMS[TRAINING_SESSION_NUMBER]["max_epsilon"]
min_epsilon   = 0.01
epsilon_decay = 500
print_interval= 20

Transition = collections.namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer():
    def __init__(self, buffer_limit=buffer_limit):
        '''
        FILL ME : This function should initialize the replay buffer `self.buffer` with maximum size of `buffer_limit` (`int`).
                  len(self.buffer) should give the current size of the buffer `self.buffer`.
        '''
        self.buffer = []
        self.max_size = buffer_limit
    
    def push(self, transition):
        '''
        FILL ME : This function should store the transition of type `Transition` to the buffer `self.buffer`.

        Input:
            * `transition` (`Transition`): tuple of a single transition (state, action, reward, next_state, done).
                                           This function might also need to handle the case  when buffer is full.

        Output:
            * None
        '''
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        '''
        FILL ME : This function should return a set of transitions of size `batch_size` sampled from `self.buffer`

        Input:
            * `batch_size` (`int`): the size of the sample.

        Output:
            * A 5-tuple (`states`, `actions`, `rewards`, `next_states`, `dones`),
                * `states`      (`torch.tensor` [batch_size, channel, height, width])
                * `actions`     (`torch.tensor` [batch_size, 1])
                * `rewards`     (`torch.tensor` [batch_size, 1])
                * `next_states` (`torch.tensor` [batch_size, channel, height, width])
                * `dones`       (`torch.tensor` [batch_size, 1])
              All `torch.tensor` (except `actions`) should have a datatype `torch.float` and resides in torch device `device`.
        '''

        mini_batch = random.sample(self.buffer, batch_size)
        
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for t in mini_batch:
            states.append(t[0])
            actions.append(t[1])
            rewards.append(t[2])
            next_states.append(t[3])
            dones.append(t[4])
        
        states = torch.tensor(states, dtype=torch.float).to(device)
        actions = torch.tensor(actions).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(device)
        dones = torch.tensor(dones, dtype=torch.float).to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        '''
        Return the length of the replay buffer.
        '''
        return len(self.buffer)

class BaseAgent(AtariDQN):
    def __init__(self, input_shape, num_actions):
        super().__init__(input_shape, num_actions)

    def act(self, state, epsilon=0.0):
        if not isinstance(state, torch.FloatTensor):
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        '''
        FILL ME : This function should return epsilon-greedy action.

        Input:
            * `state` (`torch.tensor` [batch_size, channel, height, width])
            * `epsilon` (`float`): the probability for epsilon-greedy

        Output: action (`Action` or `int`): representing the action to be taken.
                if action is of type `int`, it should be less than `self.num_actions`
        '''
        Q_values = super().forward(state)
        action = torch.argmax(Q_values)
        random_number = random.random()
        if random_number > epsilon:
            return int(action)
        else:
            action = random.randrange(self.num_actions)
            return action

def compute_loss(model, target, states, actions, rewards, next_states, dones):
    '''
    FILL ME : This function should compute the DQN loss function for a batch of experiences.

    Input:
        * `model`       : model network to optimize
        * `target`      : target network
        * `states`      (`torch.tensor` [batch_size, channel, height, width])
        * `actions`     (`torch.tensor` [batch_size, 1])
        * `rewards`     (`torch.tensor` [batch_size, 1])
        * `next_states` (`torch.tensor` [batch_size, channel, height, width])
        * `dones`       (`torch.tensor` [batch_size, 1])

    Output: scalar representing the loss.

    References:
        * MSE Loss  : https://pytorch.org/docs/stable/nn.html#torch.nn.MSELoss
        * Huber Loss: https://pytorch.org/docs/stable/nn.html#torch.nn.SmoothL1Loss
    '''

    # Input:
    Q_values_intermediate = model(states.float())
    Q_values = []
    for i in range(len(actions)):
        Q_values.append(Q_values_intermediate[i][actions[i][0]])
    Q_values = torch.stack(Q_values)

    # Target:
    Q_hat_values_intermediate = target(next_states.float()).detach()
    target_values = []
    for i in range(len(Q_hat_values_intermediate)):
        max_Q_value = torch.max(Q_hat_values_intermediate[i])
        discounted_utility = 0
        if int(dones[i][0]) == 0:
            discounted_utility = rewards[i][0] + gamma * max_Q_value
        else:
            discounted_utility = rewards[i][0]
        target_values.append(discounted_utility)
    target_values = torch.tensor(target_values)

    loss = nn.SmoothL1Loss()
    output = loss(Q_values, target_values)
    return output

def optimize(model, target, memory, optimizer):
    '''
    Optimize the model for a sampled batch with a length of `batch_size`
    '''
    batch = memory.sample(batch_size)
    loss = compute_loss(model, target, *batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def compute_epsilon(episode):
    '''
    Compute epsilon used for epsilon-greedy exploration
    '''
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * math.exp(-1. * episode / epsilon_decay)
    return epsilon

def train(model_class, env):
    '''
    Train a model of instance `model_class` on environment `env` (`GridDrivingEnv`).
    
    It runs the model for `max_episodes` times to collect experiences (`Transition`)
    and store it in the `ReplayBuffer`. It collects an experience by selecting an action
    using the `model.act` function and apply it to the environment, through `env.step`.
    After every episode, it will train the model for `train_steps` times using the 
    `optimize` function.

    Output: `model`: the trained model.
    '''

    # Initialize model and target network
    # model is loaded from saved file except first run
    loaded_model_class, model_state_dict, input_shape, num_actions = torch.load(previous_model_path)
    model = eval(loaded_model_class)(input_shape, num_actions).to(device)
    model.load_state_dict(model_state_dict)

    # NOTE: uncomment this if it is the first run
    # model = model_class(env.observation_space.shape, env.action_space.n).to(device)

    target = model_class(env.observation_space.shape, env.action_space.n).to(device)
    target.load_state_dict(model.state_dict())
    target.eval()

    # Initialize replay buffer
    memory = ReplayBuffer()

    print(model)

    # Initialize rewards, losses, and optimizer
    rewards = []
    losses = []
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for episode in range(max_episodes):
        # NOTE: get a different env each time if `randomise_env` is switched to True
        if randomise_env:
            env = get_env()

        epsilon = compute_epsilon(episode)
        state = env.reset()
        episode_rewards = 0.0

        for t in range(t_max):
            # Model takes action
            action = model.act(state, epsilon)

            # calculate modified reward matrix
            current_state = env.state
            agent_x = -1
            agent_y = -1
            for y in range(10):
                for x in range(50):
                    if current_state[1][y][x] > 0:
                        agent_x = x
                        agent_y = y
            
            mahattan_distance_reward = -1 * (agent_x + agent_y)
            occupancy_trails = current_state[3]
            number_of_occupied_neighbours = 0
            if agent_x - 1 >= 0 and occupancy_trails[agent_y][agent_x - 1] == 1:
                number_of_occupied_neighbours += 1
            if agent_x + 1 < 50 and occupancy_trails[agent_y][agent_x + 1] == 1:
                number_of_occupied_neighbours += 1
            if agent_y - 1 >= 0 and occupancy_trails[agent_y - 1][agent_x] == 1:
                number_of_occupied_neighbours += 1

            modified_reward = 0
            if with_reward_shaping:
                modified_reward = mahattan_distance_reward + (number_of_occupied_neighbours * -5)
                modified_reward = modified_reward * 0.05

            # Apply the action to the environment
            next_state, reward, done, info = env.step(action)

            # Save transition to replay buffer
            # memory.push(Transition(state, [action], [reward], next_state, [done]))
            memory.push(Transition(state, [action], [modified_reward + reward], next_state, [done]))
            # print("reward: ", final_reward + reward)

            state = next_state
            episode_rewards += reward
            if done:
                break
        rewards.append(episode_rewards)
        
        # Train the model if memory is sufficient
        if len(memory) > min_buffer:
            # if np.mean(rewards[print_interval:]) < 0.1:
            #     print('Bad initialization. Please restart the training.')
            #     exit()
            for i in range(train_steps):
                loss = optimize(model, target, memory, optimizer)
                losses.append(loss.item())

        # Update target network every once in a while
        if episode % target_update == 0:
            target.load_state_dict(model.state_dict())

        if episode % print_interval == 0 and episode > 0:
            print("[Episode {}]\tavg rewards : {:.3f},\tavg loss: : {:.6f},\tbuffer size : {},\tepsilon : {:.1f}%".format(
                            episode, np.mean(rewards[print_interval:]), np.mean(losses[print_interval*10:]), len(memory), epsilon*100))
    return model

def get_model():
    '''
    Load `model` from disk. Location is specified in `model_path`. 
    '''
    model_class, model_state_dict, input_shape, num_actions = torch.load(previous_model_path)
    model = eval(model_class)(input_shape, num_actions).to(device)
    model.load_state_dict(model_state_dict)
    return model

def save_model(model):
    '''
    Save `model` to disk. Location is specified in `model_path`. 
    '''
    data = (model.__class__.__name__, model.state_dict(), model.input_shape, model.num_actions)
    torch.save(data, new_model_path)

def get_env():
    '''
    Get the sample test cases for training and testing.
    '''
    if is_easy_env:
        return construct_easy_task_env(num_cars_per_lane)
    else:
        return construct_task_env()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train and test DQN agent.')
    parser.add_argument('--train', dest='train', action='store_true', help='train the agent')
    args = parser.parse_args()

    env = get_env()

    if args.train:
        model = train(BaseAgent, env)
        save_model(model)
    else:
        model = get_model()
    # test(model, env, max_episodes=600)