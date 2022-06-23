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
PATH = f'{cur_abs}/team2/model_weights/model_chainMDP_DQN.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class agent():
    
    def __init__(self, nState=10):
        self.input_size = nState # chain MDP 길이(state 개수)

        self.capacity = 20000 # replay memory 크기
        self.batch_size = 512
        self.training_episode = 1000 # training에 사용할 episode 개수
        self.gamma = 1.0
        self.epsilon = 1.0
        self.learning_rate = 0.001
        self.target_update_frequency = 3 # episode 단위

        self.policy_network = Q_network(self.input_size) 
        self.target_network = Q_network(self.input_size)
        self.replay_memory = ReplayMemory(self.capacity, self.batch_size)

        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)

    def save_model(self):
        torch.save(self.policy_network.state_dict(), PATH)
    
    def load_weights(self):
        self.policy_network.load_state_dict(torch.load(PATH))
        """
        test시 agent가 action을 취할 때 k개 Q network를 앙상블하기 위해 agent.k = None으로 두는 코드를 추가하였습니다.
        허용되지 않을 경우 아래 코드를 주석처리하고,  interaction.py의 calculate_performance 함수에서 주석처리된 부분을 푸시면 됩니다. 
        """
        self.epsilon = 0

    def action(self, state):
        output = self.policy_network(state)
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

    
class Q_network(nn.Module):

    def __init__(self, input_size):
        super(Q_network, self).__init__()

        self.input_size = input_size

        self.fc1 = nn.Linear(self.input_size, 10)
        self.fc2 = nn.Linear(10, 20)
        self.fc3 = nn.Linear(20, 2)

    def forward(self, x):

        if type(x) is np.ndarray:
            x = torch.tensor(x, dtype=torch.float32).to(device)
        else:
            x = x.clone().detach().to(device)
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

        return torch.tensor(np.array(s_lst), dtype=torch.float), torch.tensor(a_lst), \
            torch.tensor(r_lst), torch.tensor(np.array(s_prime_lst), dtype=torch.float), \
            torch.tensor(done_lst)
    
    def len(self):
        return len(self.memory)


#####################################################
# 모델 저장 시 실행
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from chain_mdp import ChainMDP

def train_agent(agent, env):

    for episode in tqdm(range(agent.training_episode)):

        s = env.reset()
        done = False

        while not done:    
            action = agent.action(s) # 선택된 Q network로부터 action selection
            ns, reward, done, _ = env.step(action)

            done_mask = 0.0 if done else 1.0

            transition = (s, action, reward, ns, done_mask)

            agent.replay_memory.put_sample(transition)

            # if agent.replay_memory.len() >= agent.batch_size :
            if episode > 50:
                agent.update()

            s = ns

        if episode % agent.target_update_frequency == 0:
            agent.update_target_network()

        
if __name__ == "__main__":
    agent = agent()
    train_agent(agent=agent, env=ChainMDP(10))
    agent.save_model()
#####################################################