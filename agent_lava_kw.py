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
import matplotlib.pyplot as plt

PATH = './team2_lava_model.pth'

class agent():
    
    def __init__(self, env, training=True):

        self.sample_actions = [3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
        
        self.env = env
        self.input_size = self.env.observation_space.n
        self.capacity = 10000 # replay memory 크기
        self.batch_size = 50 
        self.epsilon = 0.5 # epsilon-greedy
        self.training_episode = 3000
        self.gamma = 1.0
        self.learning_rate = 0.1
        self.target_update_frequency = 5 # episode 단위

        self.nA = env.nA
        self.nS = 60 # env.nS # 50,60 체크 해볼 것
        self.num_sa = np.ones((self.nS,self.nA)) 

        self.policy_network = Q_network(self.input_size)
        self.target_network = Q_network(self.input_size)
        self.replay_memory = ReplayMemory(self.capacity, self.batch_size)

        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)

        if training:
            self.AUC = self.train_agent()
            self.save_model(PATH)
        else:
            self.policy_network.load_state_dict(torch.load(PATH))

    def save_model(self, path):
        # if not os.path.exists(path):
        #     os.makedirs(path)
        torch.save(self.policy_network.state_dict(), path)



    def exp_bonus(self,state,action):
        return np.sqrt(2*np.log(self.num_sa.sum())/self.num_sa[np.where(state==True)[0][0],action])
        # print('state, action ',state, action )
        # print('np.where(state==True)[0][0]',np.where(state==True)[0][0])
        # print('self.num_sa[np.where(state==True)[0][0],action]',self.num_sa[np.where(state==True)[0][0],action])
        # print('2*np.log(self.num_sa.sum())/self.num_sa[np.where(state==True)[0][0],action]',2*np.log(self.num_sa.sum())/self.num_sa[np.where(state==True)[0][0],action])
        # print('np.sqrt(2*np.log(self.num_sa.sum())/self.num_sa[np.where(state==True)[0][0],action])',np.sqrt(2*np.log(self.num_sa.sum())/self.num_sa[np.where(state==True)[0][0],action]))

    def testaction(self, state):
        # print(state)
        output = self.policy_network(state)
        # print('output',output)
        # bs = torch.tensor([self.exp_bonus(state,a) for a in range(self.nA)])
        # print('bs',bs)
        action = int(torch.argmax(output))

        return action

    def action(self, state, action):
        # print(state)
        output = self.policy_network(state)
        # print('output',output)
        bs = torch.tensor([self.exp_bonus(state,a) for a in range(self.nA)])
        print('bs',bs)
        print((output + bs))
        print(torch.argmax(output + bs))
        print(int(torch.argmax(output + bs)))
        action = int(torch.argmax(output + bs))
        import sys
        sys.stdout()
        return action

    def update(self):

        s, a, r, ns, done_mask = self.replay_memory.get_samples()

        q_out = self.policy_network(s)
        q_a = q_out.gather(1,a)

        max_q_prime = self.target_network(ns).max(1)[0].unsqueeze(1)
        target = r + self.gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_target_network(self):

        self.target_network.load_state_dict(self.policy_network.state_dict())

    def train_agent(self):
        """
        agent class가 호출되면 자동으로 실행되는 함수
        agent를 학습시키고, 학습 시 sample efficiency(AUC)를 return
        """

        num_episode = self.training_episode # agent 학습시 사용할 episode 수
        cum_reward_list = list()
        for episode in tqdm(range(num_episode)):

            s = self.env.reset()
            action = 0
            s = np.eye(self.env.observation_space.n)[s]
            done = False
            cum_reward = 0.0

            while not done:    
                # print(s)
                # epsilon-greedy
                # coin = random.random()
                # if coin < self.epsilon:
                #     # print('\nrandom action\n')
                #     action = random.randint(0,self.env.action_space.n-1)
                # else:
                #     # print('\ngreedy action\n')
                action = self.action(s,action)

                self.num_sa[np.where(s==True)[0][0],action] += 1

                ns, reward, done, _ = self.env.step(action)

                done_mask = 0.0 if done else 1.0

                transition = (s, action, reward, ns, done_mask)

                self.replay_memory.put_sample(transition)

                if episode > 200:
                    self.update()

                s = ns
                cum_reward += reward
                # if done:
                #     print('episode 종료')
            
            # self.epsilon = max(self.epsilon-1/2500, 0)

            if episode % self.target_update_frequency == 0:
                self.update_target_network()

            cum_reward_list.append(cum_reward)
            AUC = np.sum(cum_reward_list)

        plt.plot(cum_reward_list)
        plt.xlabel("Episode")
        plt.ylabel("reward")
        plt.title("lava_reward")
        plt.savefig("lava_reward")
        plt.show()
        plt.close()

        return AUC            


class Q_network(nn.Module):
    
    def __init__(self, input_size):
        super(Q_network, self).__init__()

        self.input_size = input_size

        self.fc1 = nn.Linear(self.input_size, 50)
        self.fc2 = nn.Linear(50, 40)
        self.fc3 = nn.Linear(40, 4)

    def forward(self, x):

        x = torch.tensor(x, dtype=torch.float32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class ReplayMemory():
    def __init__(self, capacity, batch_size):
        self.memory = deque([], maxlen=capacity)
        self.batch_size = batch_size

    def put_sample(self, transition):
        self.memory.append(transition)

    def get_samples(self):
        """
        replay memory에서 batchsize 만큼의 (s,a,r,s',done_mask,bootstrap_mask)을 return
        """
        mini_batch = random.sample(self.memory, self.batch_size)
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
            torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
            torch.tensor(done_lst)
