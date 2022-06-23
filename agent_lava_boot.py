from ensurepip import bootstrap
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import os

from collections import deque
from collections import Counter
from tqdm import tqdm

PATH = './team2_lava_bootstrappedDQN_model.pth'

class agent():
    
    def __init__(self, env, training=True):
        self.env = env
        self.k = 5 # bootstrapped DQN에서 사용하는 Q network 개수
        self.input_size = self.env.observation_space.n # chain MDP 길이(state 개수)

        self.capacity = 300000 # replay memory 크기
        self.batch_size = 512 
        self.training_episode = 3000 # training에 사용할 episode 개수
        self.gamma = 1.0
        self.learning_rate = 0.001
        self.target_update_frequency = 2 # episode 단위

        self.policy_boot_network = Bootstrapped_network(self.k, self.input_size) 
        self.target_boot_network = Bootstrapped_network(self.k, self.input_size)
        self.replay_memory = ReplayMemory(self.capacity, self.batch_size)

        self.optimizer_list = [optim.Adam(self.policy_boot_network.network_list[i].parameters(), lr=self.learning_rate) for i in range(self.k)]

        if training:
            self.AUC, self.goal_count = self.train_agent() # 학습된 weight가 입력되지 않으면 agent 학습 및 AUC 저장
            self.save_model(PATH)
        else:
            self.policy_boot_network.load_state_dict(torch.load(PATH))

    def save_model(self, path):
        # if not os.path.exists(path):
        #     os.makedirs(path)
        torch.save(self.policy_boot_network.state_dict(), path)

    def action(self, state, k=None):
        """
        1. network 번호가 입력되지 않으면, 모든 네트워크의 argmax(output) 중 최빈값을 action으로 return (앙상블 느낌)
        2. network 번호가 입력되면, 해당 네트워크의 argmax(output)을 action으로 return

        output: action (0,1,2,3)
        """
        if k == None: # 1
            output_list = self.policy_boot_network._heads(state)
            action_list = list()
            for i in range(len(output_list)):
                action_list.append(int(torch.argmax(output_list[i])))
            c = Counter(action_list)
            return c.most_common(1)[0][0]
        else: # 2
            return int(torch.argmax(self.policy_boot_network(state, k)))
    
    def update(self):
        """
        network를 update하는 함수

        각 transition data에서 bootstrap_mask: binary vector of size k
        즉 각 transition data에서 bootstrap_mask의 i번째 원소가 1인 경우에만 해당 data를 이용하여 network i를 업데이트
            -> mini batch 안에서, bootstrap_mask의 i번째 원소가 1인 transition data들로만 mean을 계산하여 loss로 정의
        """
        s, a, r, ns, done_mask, bootstrap_mask = self.replay_memory.get_samples() # minibatch 추출

        for i in range(self.k): # 각각의 network에 대해 개별적으로 업데이트
            num_used_transition = torch.sum(bootstrap_mask[:,i]) # minibatch 안에서 network i를 업데이트하는데 사용될 transition data 개수

            q_out = self.policy_boot_network(s,i) # self.policy_network_list[i](s): i번째 policy network에 대한 batch output
            q_a = q_out.gather(1,a)

            max_q_prime = self.target_boot_network(ns,i).max(1)[0].unsqueeze(1)
            target = (r + self.gamma * max_q_prime * done_mask)
            
            # bootstrap_mask가 1인 data만으로 loss 계산
            loss = F.smooth_l1_loss(q_a, target, reduction='none')
            loss = loss * bootstrap_mask[:,i]
            loss = torch.sum(loss/num_used_transition)

            # network i 업데이트
            self.optimizer_list[i].zero_grad()
            loss.backward()
            self.optimizer_list[i].step()
    
    def update_target_network(self):
        """
        각 policy network로부터 target network 복사
        """
        for i in range(self.k):
            self.target_boot_network.network_list[i].load_state_dict(self.policy_boot_network.network_list[i].state_dict())

    def train_agent(self):
        """
        agent class가 호출되면 자동으로 실행되는 함수
        agent를 학습시키고, 학습 시 sample efficiency(AUC)를 return
        """
        # training_agent = agent(self.env)
        num_episode = self.training_episode # agent 학습시 사용할 episode 수
        goal_count = np.ones(self.k)
        cum_reward_list = list()
        for episode in tqdm(range(num_episode)):

            k = np.random.randint(self.k) # 이번 episode를 생성하는데 사용할 network 선택

            s = self.env.reset()
            s = np.eye(self.env.observation_space.n)[s]

            done = False
            cum_reward = 0.0

            while not done:    
                action = self.action(s,k) # 선택된 Q network로부터 action selection
                ns, reward, done, _ = self.env.step(action)
                if reward > 0:
                    goal_count[k] +=1

                done_mask = 0.0 if done else 1.0
                bootstrap_mask = np.random.binomial(1, 0.5, self.k)
                """
                done_mask: episode 종료면 0, 아니면 1
                bootstrap_mask: 해당 transition data를 각 Q network 업데이트에 사용할지 안할지 여부
                    ex) bootstrap_mask가 [1,0,0,1,1]이면, 해당 transition 데이터는 1,4,5번 network 업데이트에만 사용됨 
                """

                transition = (s, action, reward, ns, done_mask, bootstrap_mask)

                self.replay_memory.put_sample(transition)

                if episode > 200:
                    self.update()

                s = ns
                cum_reward += reward

            if episode % self.target_update_frequency == 0:
                self.update_target_network()

            cum_reward_list.append(cum_reward)
            AUC = np.sum(cum_reward_list)
    
        return AUC, goal_count           

    
class Q_network(nn.Module):

    def __init__(self, input_size):
        super(Q_network, self).__init__()

        self.input_size = input_size

        self.fc1 = nn.Linear(self.input_size, 40)
        self.fc2 = nn.Linear(40, 20)
        self.fc3 = nn.Linear(20, 4)

    def forward(self, x):

        x = torch.tensor(x, dtype=torch.float32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class Bootstrapped_network(nn.Module):
    """
    Q network k개
    """

    def __init__(self, k, input_size):
        super(Bootstrapped_network, self).__init__()

        self.network_list = nn.ModuleList([Q_network(input_size) for i in range(k)])

    def _heads(self, x):
        """
        k개 Q network 각각에 input x(state)를 입력한 값들을 list로 반환
        """
        return [network(x) for network in self.network_list]

    def forward(self, x, k):
        """
        k번째 Q network에 input x를 입력한 값을 반환
        """
        return self.network_list[k](x)


class ReplayMemory():
    def __init__(self, capacity, batch_size):
        self.memory = deque([], maxlen=capacity)
        self.batch_size = batch_size
        # self.network_num = k

    def put_sample(self, transition):
        self.memory.append(transition)

    def get_samples(self):
        """
        replay memory에서 batchsize 만큼의 (s,a,r,s',done_mask,bootstrap_mask)을 return
        """
        mini_batch = random.sample(self.memory, self.batch_size)
        s_lst, a_lst, r_lst, s_prime_lst, done_lst, bootstrap_lst = [], [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask, bootstrap_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_lst.append([done_mask])
            bootstrap_lst.append(bootstrap_mask)

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
            torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
            torch.tensor(done_lst), torch.tensor(bootstrap_lst)
