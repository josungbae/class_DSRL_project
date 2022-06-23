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
    
    def __init__(self, **kwargs):

        # self.sample_actions = [3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]

        self.env = kwargs.get('env')
        self.training = kwargs.get('training') # training 실행 여부 (false일 경우 학습된 model을 통해 test)
        self.input_size = self.env.observation_space.n
        self.capacity = 300000 # replay memory 크기
        self.batch_size = 512
        self.epsilon = 1.0 # epsilon-greedy
        self.training_episode = 300
        self.gamma = 1.0
        self.learning_rate = 0.001
        self.target_update_frequency = 2 # episode 단위
        self.num_sa = np.ones((self.env.observation_space.n,self.env.action_space.n)) 

        self.policy_network = Q_network(self.input_size)
        self.target_network = Q_network(self.input_size)
        self.replay_memory = ReplayMemory(self.capacity, self.batch_size)

        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)

        if self.training:
            self.AUC, self.goal_count, self.cum_reward_list = self.train_agent()
            self.save_model(PATH)
        else:
            self.policy_network.load_state_dict(torch.load(PATH))

    def save_model(self, path):
        # if not os.path.exists(path):
        #     os.makedirs(path)
        torch.save(self.policy_network.state_dict(), path)

    def exp_bonus(self,state,action):
        return np.sqrt(2*np.log(self.num_sa.sum())/self.num_sa[np.where(state==True)[0][0],action])

    def action(self, state):
        # print(state)
        output = self.policy_network(state)
        # bs = torch.tensor([self.exp_bonus(state,a) for a in range(self.env.action_space.n)])

        action = int(torch.argmax(output))

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
        goal_count = 0

        for episode in tqdm(range(num_episode)):
            step = 0
            s = self.env.reset()
            s = np.eye(self.env.observation_space.n)[s]
            done = False
            cum_reward = 0.0

            while not done:
                # print('state:',np.unravel_index(np.where(s==True)[0][0],(6,10)))
                # epsilon-greedy
                coin = random.random()
                if coin < self.epsilon:
                    # print('\nrandom action\n')
                    action = random.randint(0,self.env.action_space.n-1)
                else:
                    # print('\ngreedy action\n')
                    action = self.action(s)
                # self.num_sa[np.where(s==True)[0][0],action] += 1
                # print(self.num_sa)

                ns, reward, done, _ = self.env.step(action)
                if reward > 0:
                    # reward = reward * 10000
                    goal_count += 1

                done_mask = 0.0 if done else 1.0

                transition = (s, action, reward, ns, done_mask)

                self.replay_memory.put_sample(transition)

                if episode > 200:
                    self.update()

                s = ns
                cum_reward += reward
                step += 1
                
            if episode > 200:
                self.epsilon = max(self.epsilon-1/300, 0)
            # if episode > 2500:
            #     self.epsilon = 0
            if episode % self.target_update_frequency == 0:
                self.update_target_network()

            cum_reward_list.append(cum_reward)
            AUC = np.sum(cum_reward_list)
    
        return AUC, goal_count, cum_reward_list            


class Q_network(nn.Module):
    
    def __init__(self, input_size):
        super(Q_network, self).__init__()

        self.input_size = input_size

        self.fc1 = nn.Linear(self.input_size, 40)
        self.fc2 = nn.Linear(40, 20)
        # self.fc3 = nn.Linear(40, 20)
        self.fc3 = nn.Linear(20, 4)

    def forward(self, x):

        x = torch.tensor(x, dtype=torch.float32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
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

# class ReplayMemory:
#     def __init__(self, capacity, batch_size):
#         self.memory = {}
#         for i in range(60):
#             self.memory.setdefault(i, {})
#             for j in range(4):
#                 self.memory[i].setdefault(j, deque([], maxlen=60))
#         self.batch_size = batch_size
#         self.visit = {}
#         for i in range(60):
#             self.visit.setdefault(i, {})
#             for j in range(4):
#                 self.visit[i].setdefault(j, 0)
#         self.total_visit = 0

#     def put_sample(self, transition):
#         state_int = np.argmax(transition[0], axis=0)
#         action = transition[1]
#         self.visit[state_int][action] += 1
#         self.total_visit += 1
#         if self.total_visit % 2000 == 0:
#             for row in range(6):
#                 print([i[0] + i[1] + i[2] + i[3] for i in self.visit.values()][row * 10 : (row + 1) * 10])
#             print()
#         self.memory[state_int][action].append(transition)
#         temp_list = []
#         for i in range(60):
#             for j in range(4):
#                 temp_list += list(self.memory[i][j])
#         self.memory_merge = deque(temp_list)

#     def get_samples(self):
#         """
#         replay memory에서 batchsize 만큼의 (s,a,r,s',done_mask,bootstrap_mask)을 return
#         """
#         mini_batch = random.sample(self.memory_merge, self.batch_size)

#         s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []

#         for transition in mini_batch:
#             s, a, r, s_prime, done_mask = transition
#             s_lst.append(s)
#             a_lst.append([a])
#             r_lst.append([r])
#             s_prime_lst.append(s_prime)
#             done_lst.append([done_mask])

#         return torch.tensor(np.array(s_lst), dtype=torch.float), torch.tensor(np.array(a_lst)), torch.tensor(np.array(r_lst)), torch.tensor(np.array(s_prime_lst), dtype=torch.float), torch.tensor(np.array(done_lst))