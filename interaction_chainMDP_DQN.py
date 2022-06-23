import numpy as np
import random

def calculate_performance(episodes, env, agent):

    episodic_returns = []

    for epi in range(episodes):
        
        s = env.reset()

        done = False
        cum_reward = 0.0

        while not done:    
            action = agent.action(s)
            ns, reward, done, _ = env.step(action)
            cum_reward += reward
            s = ns
        
        episodic_returns.append(cum_reward)
                    
    return np.sum(episodic_returns)

def calculate_sample_efficiency(episodes, env, agent):

    episodic_returns = []
    
    for epi in range(episodes):
        
        s = env.reset()

        done = False
        cum_reward = 0.0
               
        while not done:    
            # epsilon-greedy
            coin = random.random()
            if coin < agent.epsilon:
                action = random.randint(0, env.action_space.n - 1)
            else:
                action = agent.action(s)
            # action = agent.action(s)
            ns, reward, done, _ = env.step(action)
            cum_reward += reward
            # s = ns
            
            #####################
            # If your agent needs to update the weights at every time step, complete your update process in this area.
            # e.g., agent.update()
            done_mask = 0.0 if done else 1.0
            transition = (s, action, reward, ns, done_mask)

            agent.replay_memory.put_sample(transition)
            
            # if agent.replay_memory.len() > agent.batch_size:
            if epi > 50:
                agent.update()
            #####################
            """
            s=ns를 replay memory 저장을 위해 아래로 이동하였습니다
            """
            s = ns 
        #####################
        # elif your agent needs to update the weights at the end of every episode, complete your update process in this area.
        # e.g., agent.update()
        if epi % agent.target_update_frequency == 0:
            agent.update_target_network()
        agent.epsilon = max(agent.epsilon - 1 / 200, 0)
        #####################
        
        episodic_returns.append(cum_reward)
                    
    return np.sum(episodic_returns)

