import gym
from lava_grid import ZigZag6x10
from agent_lava import agent
import random
import numpy as np

# default setting
max_steps = 100
stochasticity = 0 # probability of the selected action failing and instead executing any of the remaining 3
no_render = True

env = ZigZag6x10(max_steps=max_steps, act_fail_prob=stochasticity, goal=(5, 9), numpy_state=False)

done = False
cum_reward = 0.0
training = True
""" Your agent"""
agent = agent(env, training=training)

s = env.reset()
s = np.eye(env.observation_space.n)[s]

# moving costs -0.01, falling in lava -1, reaching goal +1
# final reward is number_of_steps / max_steps
while not done:
    if training :     
        action = agent.action(s) 
    else : 
        action = agent.testaction(s) 

    ns, reward, done, _ = env.step(action)

    s = ns
    cum_reward += reward

print(f"total reward: {cum_reward}")
