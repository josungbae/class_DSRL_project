# %%
import gym
from lava_grid import ZigZag6x10
# from agent_lava_hDQN import agent
from agent_lava import agent
import random
import numpy as np
import matplotlib.pyplot as plt
import collections

# default setting
max_steps = 100
stochasticity = 0 # probability of the selected action failing and instead executing any of the remaining 3
no_render = True

env = ZigZag6x10(max_steps=max_steps, act_fail_prob=stochasticity, goal=(5, 9), numpy_state=False)

""" Your agent"""
agent = agent(env=env, training=False)

# print('training 중 goal 도달 횟수:', agent.goal_count)
# print('training 중 controller 성공 횟수:', np.sum(agent.controller_success))
# print('방문 횟수 종류:',collections.Counter(np.ravel(agent.num_sa, order='C')))

s = env.reset()
s = np.eye(env.observation_space.n)[s]
done = False
cum_reward = 0.0

# moving costs -0.01, falling in lava -1, reaching goal +1
# final reward is number_of_steps / max_steps

# partial_path = [3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2]
# for action in partial_path:
#     print('\nstate:',s)
#     print('o-action:',action)
#     ns, reward, done, _ = env.step(action)
#     s = ns

while not done:
    print('\nstate:',np.unravel_index(np.where(s==True)[0][0],(6,10)))
    action = agent.action(s)

    print('action:',action)
    ns, reward, done, _ = env.step(action)
    print('reward:',reward)

    cum_reward += reward

    s = ns

print(f"total reward: {cum_reward}")
print(f"AUC: {agent.AUC}")

# np.set_printoptions(precision=6, suppress=True)
# print('Q table:',agent.Qtable)
# font = {'color': 'w'}
# plt.plot(agent.cum_reward_list)
# plt.xlabel('episodes', fontdict=font)
# plt.ylabel('total rewards', fontdict=font)
# plt.xticks(color = 'white')
# plt.yticks(color = 'white')
# plt.show()
# %%