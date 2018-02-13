
import numpy as np
import random

ENV_SIZE = 0

#random.seed(1234)
#np.random.seed(1234)
#1 Is Agent
#2 Is Goal
#3 Is blocked tile
#4 Is Treat

def set_up_environment(size = 5):
    global ENV_SIZE
    ENV_SIZE = size
    environment = np.zeros((size, size), dtype = int)

    agent_pos = [random.randint(0, size - 1), random.randint(0, size - 1)]
    environment[agent_pos[0], agent_pos[1]] = 1

    goal_pos = [random.randint(0, size - 1), random.randint(0, size - 1)]
    while(environment[goal_pos[0]][goal_pos[1]] == 1):
        goal_pos = [random.randint(0, size - 1), random.randint(0, size - 1)]

    environment[goal_pos[0], goal_pos[1]] = 2

    number_of_obstacles = (random.randint(1, int(np.ceil(size * size)/10)))
    print(number_of_obstacles)
    for i in range(number_of_obstacles):
        obstacle_pos = [random.randint(0, size - 1), random.randint(0, size - 1)]
        
        while(environment[obstacle_pos[0]][obstacle_pos[1]] == 1 or 
              environment[obstacle_pos[0]][obstacle_pos[1]] == 2 or
              environment[obstacle_pos[0]][obstacle_pos[1]] == 3 
              ):
            obstacle_pos = [random.randint(0, size - 1), random.randint(0, size - 1)]
        environment[obstacle_pos[0]][obstacle_pos[1]] = 3

    return environment

def set_rewards():
    return {"goal" : 1, "treat" : -1, "blank" : 0, "agent" : 0, "blocked" : 0}
    
def reached_goal(goal_pos, last_pos):
    if(goal_pos == last_pos).all():
        return True

    return False

def get_reward_matrix(environment):
    rewards = np.zeros((environment.shape[0], environment.shape[1]))
    values = set_rewards()
    for i in range(environment.shape[0]):
        for j in range(environment.shape[1]):
            if environment[i, j] == 0:
                rewards[i, j] = values["blank"]
            elif environment[i, j] == 1:
                rewards[i, j] = values["agent"]
            elif environment[i, j] == 2:
                rewards[i, j] = values["goal"]
            elif environment[i, j] == 3:
                rewards[i, j] = values["blocked"]
            elif environment[i, j] == 4:
                rewards[i, j] = values["treat"]
    return rewards


def get_agent_pos(environment):
    for i in range(environment.shape[0]):
        for j in range(environment.shape[1]):
            if environment[i][j] == 1:
                return np.array([i, j])

def get_goal_pos(environment):
    for i in range(environment.shape[0]):
        for j in range(environment.shape[1]):
            if environment[i][j] == 2:
                return np.array([i, j])

def can_move(position, action, env):
    if action == 0:
        next_pos = [position[0] - 1, position[1]]
    elif action == 1:
        next_pos = [position[0] + 1, position[1]]
    elif action == 2:
        next_pos = [position[0], position[1] - 1]
    elif action == 3:
        next_pos = [position[0], position[1] + 1]
  
    if (
        next_pos[0] >= 0 and 
        next_pos[0] < ENV_SIZE and 
        next_pos[1] >= 0 and 
        next_pos[1] < ENV_SIZE and
        env[next_pos[0], next_pos[1]] != 3):
        return True
    
    return False


def step(action, position, reward_values, action_space):
    #0: up
    #1: down
    #2: left
    #3: right

    if action == 0:
        next_pos = [position[0] - 1, position[1]]
        next_state = get_state(next_pos[0], next_pos[1], action_space - 1)
    elif action == 1:
        next_pos = [position[0] + 1, position[1]]
        next_state = get_state(next_pos[0], next_pos[1], action_space - 1)
    elif action == 2:
        next_pos = [position[0], position[1] - 1]
        next_state = get_state(next_pos[0], next_pos[1], action_space - 1)
    elif action == 3:
        next_pos = [position[0], position[1] + 1]
        next_state = get_state(next_pos[0], next_pos[1], action_space - 1)
    
    reward = reward_values[next_pos[0], next_pos[1]]

    return next_state, reward, next_pos


def get_state(posI, posJ, maxJ):
    return posI * (maxJ + 1) + posJ
      
def updateEnv(last_pos, next_pos, env):
    env[last_pos[0], last_pos[1]] = 0
    env[next_pos[0], next_pos[1]] = 1
    return env

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
def q_learning(gamma, alpha, episodes, environment_size):
    epsilon = 0.99
    epsilon_min_value = 0.1
    
    last_pos = np.array([[-1], [-1]], dtype = int)
    action_space = 4
    Q = np.zeros((environment_size * environment_size, action_space), dtype = float)
    current_reward = .0
    

    original_environment = set_up_environment(environment_size)
    reward_values = get_reward_matrix(original_environment)
    print("El nuevo ambiente es: \n", original_environment)
    input("Presione enter para iniciar")

    for i in range(episodes):
        #reset the env
        environment = original_environment.copy()
        
        epsilon = 1 / (i + 1)
        if(epsilon <= epsilon_min_value):
            epsilon = epsilon_min_value
        
        last_pos = get_agent_pos(environment)
        goal_pos = get_goal_pos(environment)
        while(not reached_goal(goal_pos, last_pos)):
            last_state = get_state(last_pos[0], last_pos[1], action_space-1)
            greed_or_not = np.random.random_sample()
            #print(greed_or_not)
            if(greed_or_not >= epsilon):
                #greed
                #print("Greeding")
                action = np.argmax(Q[last_state, :])
            else:
                #explore
                #print("Exploring")
                action = np.random.randint(0, action_space)
          
            while not can_move(last_pos, action, environment):
                if(greed_or_not <= epsilon):
                    #greed
                    #print("Greeding")
                    aux_Q = Q.copy()
                    aux_Q[last_state, action] = -100000
                    action = np.argmax(aux_Q[last_state, :])

                    if aux_Q[last_state, : ].all() == -100000:
                        greed_or_not = 1 - epsilon
                else:
                    #explore
                    #print("Exploring")
                    action = np.random.randint(0, action_space)

            next_state, reward, next_pos = step(action, last_pos, reward_values, action_space)

            Q[last_state, action] = Q[last_state, action] + alpha * (reward + gamma * np.max(Q[next_state, : ]) - Q[last_state, action])
            current_reward += reward
            #print(updateEnv(last_pos, next_pos, environment))
            last_pos = next_pos
            last_state = next_state
            
            #input("Mover al jugador\n")
        if i % 100 == 0:
            print("\nEpisode: {0} \nTotal reward: {1} \nEpsilon: {2}".format(i, current_reward, epsilon))
            print(np.around(Q, decimals=2))
        
        current_reward = 0


def main():
    q_learning(0.1, 0.9, 2000, 6)

if __name__ == "__main__":
    main()