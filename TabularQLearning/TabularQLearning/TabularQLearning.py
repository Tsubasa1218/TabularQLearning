
import tensorflow as tf
import numpy as np
import random

#1 Is Agent
#2 Is Goal
#3 Is blocked tile
#4 Is Treat

def set_up_environment(size = 5):
    environment = np.zeros((size, size), dtype = int)

    agent_pos = [random.randint(0, size - 1), random.randint(0, size - 1)]

    environment[agent_pos[0]][agent_pos[1]] = 1

    goal_pos = [random.randint(0, size - 1), random.randint(0, size - 1)]

    while(environment[goal_pos[0]][goal_pos[1]] != 1):
        goal_pos = [random.randint(0, size - 1), random.randint(0, size - 1)]

    environment[goal_pos[0]][goal_pos[1]] = 2

    number_of_obstacles = random.randint(1, size ** 2) / 10

    for i in range(number_of_obstacles):
        obstacle_pos = [random.randint(0, size - 1), random.randint(0, size - 1)]
        
        while(environment[obstacle_pos[0]][obstacle_pos[1]] != 1 or 
              environment[obstacle_pos[0]][obstacle_pos[1]] != 2 or
              environment[obstacle_pos[0]][obstacle_pos[1]] != 3 
              ):
            environment[obstacle_pos[0]][obstacle_pos[1]] = 3

    return environment


#The Q-Learning algorithm goes as follows:
#    1. Set the gamma parameter, and environment rewards in matrix R.
#    2. Initialize matrix Q to zero.
#    3. For each episode:
#        Select a random initial state.
#        Do While the goal state hasn't been reached.
#            Select one among all possible actions for the current state.
#            Using this possible action, consider going to the next state.
#            Get maximum Q value for this next state based on all possible actions.
#            Compute: Q(state, action) = R(state, action) + Gamma * Max[Q(next state, all actions)]
#            Set the next state as the current state.
#        End Do
#    End For

def set_rewards(environment):
    return {"2" : 1, "4" : -1, "0" : 0}
    
def reached_goal(environment, last_pos):
    if(environment[last_pos[0]][last_pos[1]] == 2):
        return True

    return False

def select_action(environment, epsilon):
    pos = np.zeros((1,2))
    for i in range(environment.shape[0]):
        for j in range(environment.shape[1]):
            if environment[i][j] == 1:
                pass
            pass
        pass


def q_learning(gamma, alpha, episodes, environment_size):
    epsilon = 1.0
    epsilon_min_value = 0.1
    reward_values = set_rewards()

    last_state = np.array([[-1], [-1]], dtype = int)

    q_table = np.zeros((environment_size, environment_size), dtype = float)

    for i in range(episodes):
        environment = set_up_environment(environment_size)

        while(not reached_goal(environment, last_pos)):




