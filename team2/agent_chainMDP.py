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

cur_abs = os.path.abspath('.')
PATH = f'{cur_abs}/team2/model_weights/model_chainMDP.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class agent():
    
    def __init__(self, nState=10, nAction=2):
        self.input_size = nState # chain MDP 길이(state 개수)
        self.output_size = nAction

        self.num_k = 7 # bootstrapped DQN에서 사용하는 Q network 개수
        self.k = None
        self.capacity = 20000 # replay memory 크기
        self.batch_size = 100
        self.training_episode = 1000 # training에 사용할 episode 개수
        self.gamma = 1.0
        self.learning_rate = 0.001
        self.target_update_frequency = 3 # episode 단위

        self.policy_boot_network = Bootstrapped_network(self.num_k, self.input_size, self.output_size) 
        self.target_boot_network = Bootstrapped_network(self.num_k, self.input_size, self.output_size)
        self.replay_memory = ReplayMemory(self.capacity, self.batch_size)

        self.optimizer_list = [optim.Adam(self.policy_boot_network.network_list[i].parameters(), lr=self.learning_rate) for i in range(self.num_k)]

    def save_model(self):
        torch.save(self.policy_boot_network.state_dict(), PATH)
    
    def load_weights(self):
        self.policy_boot_network.load_state_dict(torch.load(PATH))
        """
        test시 agent가 action을 취할 때 k개 Q network를 앙상블하기 위해 agent.k = None으로 두는 코드를 추가하였습니다.
        허용되지 않을 경우 아래 코드를 주석처리하고,  interaction.py의 calculate_performance 함수에서 주석처리된 부분을 푸시면 됩니다. 
        """
        self.k = None

    def action(self, state):
        """
        1. network 번호가 입력되지 않으면, 모든 네트워크의 argmax(output) 중 최빈값을 action으로 return (앙상블)
        2. network 번호가 입력되면, 해당 네트워크의 argmax(output)을 action으로 return
        """
        if self.k == None: # 1
            output_list = self.policy_boot_network._heads(state)
            action_list = list()
            for i in range(len(output_list)):
                action_list.append(int(torch.argmax(output_list[i])))
            c = Counter(action_list)
            return c.most_common(1)[0][0]
        else: # 2
            # print(f'k:{self.k}')
            # print(f'output:{self.policy_boot_network(state, self.k)}')
            return int(torch.argmax(self.policy_boot_network(state, self.k)))
    
    def update(self):
        """
        network를 update하는 함수

        각 transition data에서 bootstrap_mask: binary vector of size k
        즉 각 transition data에서 bootstrap_mask의 i번째 원소가 1인 경우에만 해당 data를 이용하여 network i를 업데이트
            -> mini batch 안에서, bootstrap_mask의 i번째 원소가 1인 transition data들로만 mean을 계산하여 loss로 정의
        """
        s, a, r, ns, done_mask, bootstrap_mask = self.replay_memory.get_samples() # minibatch 추출

        for i in range(self.num_k): # 각각의 network에 대해 개별적으로 업데이트
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
        for i in range(self.num_k):
            self.target_boot_network.network_list[i].load_state_dict(self.policy_boot_network.network_list[i].state_dict())

    
class Q_network(nn.Module):

    def __init__(self, input_size, output_size):
        super(Q_network, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.fc1 = nn.Linear(self.input_size, 10)
        self.fc2 = nn.Linear(10, 20)
        self.fc3 = nn.Linear(20, self.output_size)
        # self.fc1 = nn.Linear(self.input_size, 20)
        # self.fc2 = nn.Linear(20, 5)
        # self.fc3 = nn.Linear(5, 2)

    def forward(self, x):

        if type(x) is np.ndarray:
            x = torch.tensor(x, dtype=torch.float32).to(device)
        else:
            x = x.clone().detach().to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class Bootstrapped_network(nn.Module):
    """
    Q network k개
    """

    def __init__(self, k, input_size, output_size):
        super(Bootstrapped_network, self).__init__()

        self.network_list = nn.ModuleList([Q_network(input_size, output_size) for i in range(k)])

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

        return torch.tensor(np.array(s_lst), dtype=torch.float), torch.tensor(a_lst), \
            torch.tensor(r_lst), torch.tensor(np.array(s_prime_lst), dtype=torch.float), \
            torch.tensor(done_lst), torch.tensor(np.array(bootstrap_lst))
    
    def len(self):
        return len(self.memory)


#####################################################
# 모델 저장 시 실행
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from chain_mdp import ChainMDP

def train_agent(agent, env):

    for episode in tqdm(range(agent.training_episode)):

        agent.k = np.random.randint(agent.num_k) # 이번 episode를 생성하는데 사용할 network 선택

        s = env.reset()
        done = False

        while not done:    
            action = agent.action(s) # 선택된 Q network로부터 action selection
            ns, reward, done, _ = env.step(action)

            done_mask = 0.0 if done else 1.0
            bootstrap_mask = np.random.binomial(1, 0.5, agent.num_k)
            """
            done_mask: episode 종료면 0, 아니면 1
            bootstrap_mask: 해당 transition data를 각 Q network 업데이트에 사용할지 안할지 여부
                ex) bootstrap_mask가 [1,0,0,1,1]이면, 해당 transition 데이터는 1,4,5번 network 업데이트에만 사용됨 
            """

            transition = (s, action, reward, ns, done_mask, bootstrap_mask)

            agent.replay_memory.put_sample(transition)

            # if agent.replay_memory.len() >= agent.batch_size :
            if episode > 10:
                agent.update()

            s = ns

        if episode % agent.target_update_frequency == 0:
            agent.update_target_network()

        
if __name__ == "__main__":
    agent = agent()
    train_agent(agent=agent, env=ChainMDP(10))
    agent.save_model()
#####################################################