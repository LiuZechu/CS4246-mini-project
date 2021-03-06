import gym
from gym.utils import seeding
from gym_grid_driving.envs.grid_driving import LaneSpec

def construct_easy_task_env(num_cars):
    config = {'observation_type': 'tensor', 'agent_speed_range': [-3, -1], 'width': 50,
              'lanes': [LaneSpec(cars=num_cars, speed_range=[-2, -1]), 
                        LaneSpec(cars=num_cars, speed_range=[-2, -1]), 
                        LaneSpec(cars=num_cars, speed_range=[-1, -1]), 
                        LaneSpec(cars=num_cars, speed_range=[-3, -1]), 
                        LaneSpec(cars=num_cars, speed_range=[-2, -1]), 
                        LaneSpec(cars=num_cars, speed_range=[-2, -1]), 
                        LaneSpec(cars=num_cars, speed_range=[-3, -2]), 
                        LaneSpec(cars=num_cars, speed_range=[-1, -1]), 
                        LaneSpec(cars=num_cars, speed_range=[-2, -1]), 
                        LaneSpec(cars=num_cars, speed_range=[-2, -2])]
            }
    return gym.make('GridDriving-v0', **config)