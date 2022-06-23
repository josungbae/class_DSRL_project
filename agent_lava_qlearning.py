from ensurepip import bootstrap
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

from collections import deque
from collections import Counter
from tqdm import tqdm

PATH = './team2_lava_model.pth'

class agent():
    
    def __init__(self, env, training=True):

        self.sample_actions = [3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
        
        self.env = env
        self.input_size = self.env.observation_space.n
        self.capacity = 10000 # replay memory 크기
        self.batch_size = 50 
        self.epsilon = 1.0 # epsilon-greedy
        self.training_episode = 10000
        self.gamma = 0.9
        self.learning_rate = 0.05
        self.target_update_frequency = 3 # episode 단위

        self.Qtable = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        # self.policy_network = Q_network(self.input_size)
        # self.target_network = Q_network(self.input_size)
        # self.replay_memory = ReplayMemory(self.capacity, self.batch_size)

        # self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)

        if training:
            self.AUC = self.train_agent()
            # self.save_model(PATH)
        else:
            return
            # self.policy_network.load_state_dict(torch.load(PATH))

    # def save_model(self, path):
    #     # if not os.path.exists(path):
    #     #     os.makedirs(path)
    #     torch.save(self.policy_network.state_dict(), path)

    def action(self, state):
        # print(state)
        # output = self.policy_network(state)
        action = np.argmax(self.Qtable[np.where(state==True)[0][0],:])

        return action

    # def update(self):

    #     s, a, r, ns, done_mask = self.replay_memory.get_samples()

    #     q_out = self.policy_network(s)
    #     q_a = q_out.gather(1,a)

    #     max_q_prime = self.target_network(ns).max(1)[0].unsqueeze(1)
    #     target = r + self.gamma * max_q_prime * done_mask
    #     loss = F.smooth_l1_loss(q_a, target)

    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
    
    # def update_target_network(self):

    #     self.target_network.load_state_dict(self.policy_network.state_dict())

    def train_agent(self):
        """
        agent class가 호출되면 자동으로 실행되는 함수
        agent를 학습시키고, 학습 시 sample efficiency(AUC)를 return
        """

        num_episode = self.training_episode # agent 학습시 사용할 episode 수
        cum_reward_list = list()
        for episode in tqdm(range(num_episode)):

            s = self.env.reset()
            s = np.eye(self.env.observation_space.n)[s]
            done = False
            cum_reward = 0.0

            while not done:    
                # print(s)
                # epsilon-greedy
                coin = random.random()
                if coin < self.epsilon:
                    # print('\nrandom action\n')
                    action = random.randint(0,self.env.action_space.n-1)
                else:
                    # print('\ngreedy action\n')
                    action = self.action(s)

                ns, reward, done, _ = self.env.step(action)

                # done_mask = 0.0 if done else 1.0
                self.Qtable[np.where(s==True)[0][0],action] \
                    = (1-self.learning_rate) * self.Qtable[np.where(s==True)[0][0],action] \
                        + self.learning_rate * (reward + self.gamma * np.max(self.Qtable[np.where(s==True)[0][0],:]))

                # transition = (s, action, reward, ns, done_mask)

                # self.replay_memory.put_sample(transition)

                # if episode > 200:
                #     self.update()

                s = ns
                cum_reward += reward
                # if done:
                #     print('episode 종료')
            
            self.epsilon = max(self.epsilon-1/10000, 0.1)

            # if episode % self.target_update_frequency == 0:
            #     self.update_target_network()

            cum_reward_list.append(cum_reward)
            AUC = np.sum(cum_reward_list)
    
        return AUC            


# class Q_network(nn.Module):
    
#     def __init__(self, input_size):
#         super(Q_network, self).__init__()

#         self.input_size = input_size

#         self.fc1 = nn.Linear(self.input_size, 50)
#         self.fc2 = nn.Linear(50, 40)
#         self.fc3 = nn.Linear(40, 4)

#     def forward(self, x):

#         x = torch.tensor(x, dtype=torch.float32)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)

#         return x
