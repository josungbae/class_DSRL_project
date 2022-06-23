from chain_mdp import ChainMDP
from agent_chainMDP import agent
from tqdm import tqdm

import numpy as np
import warnings

warnings.filterwarnings('ignore')


# receive 1 at rightmost state and receive small reward at leftmost state
env = ChainMDP(10)

# # Training
# def train_agent(env):
#     training_agent = agent(env)
#     num_episode = 1000 # agent 학습시 사용할 episode 수

#     cum_reward_list = list()
#     for episode in tqdm(range(num_episode)):

#         k = np.random.randint(training_agent.k) # 이번 episode를 생성하는데 사용할 network 선택

#         s = env.reset()
#         done = False
#         cum_reward = 0.0

#         while not done:    
#             action = training_agent.action(s,k) # 선택된 Q network로부터 action selection
#             ns, reward, done, _ = env.step(action)

#             done_mask = 0.0 if done else 1.0
#             bootstrap_mask = np.random.binomial(1, 0.5, training_agent.k)
#             """
#             done_mask: episode 종료면 0, 아니면 1
#             bootstrap_mask: 해당 transition data를 각 Q network 업데이트에 사용할지 안할지 여부
#                 ex) bootstrap_mask가 [1,0,0,1,1]이면, 해당 transition 데이터는 1,4,5번 network 업데이트에만 사용됨 
#             """

#             transition = (s, action, reward, ns, done_mask, bootstrap_mask)
            
#             training_agent.replay_memory.put_sample(transition)

#             if episode > 50:
#                 training_agent.update()

#             s = ns
#             cum_reward += reward

#         if episode % training_agent.target_update_frequency == 0:
#             training_agent.update_target_network()

#         cum_reward_list.append(cum_reward)
#         AUC = np.sum(cum_reward_list)
    
#     return training_agent, AUC


# Test
total_reward_list = list()
AUC_list = list()
for episode in range(20): # test하는데 사용할 episode 개수. ex) 20이면 총 20번 학습하고, 각각에 대해 test. 빨리 끝내고 싶으면 1로 하세요.

    trained_agent = agent(env)
    AUC = trained_agent.AUC # agent를 학습하고, 학습된 agent를 trained_agent로 정의
    s = env.reset()
    done = False
    cum_reward = 0.0

    while not done:    
        action = trained_agent.action(s)
        ns, reward, done, _ = env.step(action)

        cum_reward += reward

        s = ns
    print(f"Episode {episode} - total reward (in test): {cum_reward}")
    print(f"Episode {episode} - sample efficiency(AUC): {AUC}")
    total_reward_list.append(cum_reward)
    AUC_list.append(AUC)

print(f"\n\nAverage total reward in test: {np.mean(total_reward_list)}")
print(F"Final sample efficiency score: {np.mean(AUC_list)}")
