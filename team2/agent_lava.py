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

cur_abs = os.path.abspath(".")
PATH = f"{cur_abs}/team2/model_weights/model_lava.pth"


class agent:
    def __init__(self, nState=60, nAction=4, batch_size=512, capacity=5000, epsilon=0.2, alpha=100, training_episode=3000):
        self.nState = nState
        self.input_size = nState
        self.output_size = nAction
        self.capacity = capacity  # replay memory 크기
        self.batch_size = batch_size
        self.epsilon = epsilon  # epsilon-greedy
        self.alpha = 100

        self.training_episode = training_episode
        self.gamma = 0.9
        self.learning_rate = 0.001
        self.target_update_frequency = 1  # episode 단위

        self.policy_network = Q_network(self.input_size, self.output_size)
        self.target_network = Q_network(self.input_size, self.output_size)
        self.replay_memory = ReplayMemory(self.capacity, self.batch_size, self.nState, self.output_size)

        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)

    def save_model(self):
        torch.save(self.policy_network.state_dict(), PATH)

    def load_weights(self):
        self.policy_network.load_state_dict(torch.load(PATH))

    def action(self, state):
        # print(state)
        output = self.policy_network(state)
        action = int(torch.argmax(output))

        return action

    def update(self):

        s, a, r, ns, done_mask = self.replay_memory.get_samples()

        q_out = self.policy_network(s)
        q_a = q_out.gather(1, a)

        max_q_prime = self.target_network(ns).max(1)[0].unsqueeze(1)
        target = r + self.gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):

        self.target_network.load_state_dict(self.policy_network.state_dict())


class Q_network(nn.Module):
    def __init__(self, input_size, output_size):
        super(Q_network, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.fc1 = nn.Linear(self.input_size, 50)
        self.fc2 = nn.Linear(50, 40)
        self.fc3 = nn.Linear(40, self.output_size)

    def forward(self, x):

        x = torch.tensor(x, dtype=torch.float32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class ReplayMemory:
    def __init__(self, capacity, batch_size, nState, nAction):
        self.nState = nState
        self.nAction = nAction
        self.total_visit = 0
        self.batch_size = batch_size
        self.memory = {}
        for i in range(self.nState):
            self.memory.setdefault(i, {})
            for j in range(self.nAction):
                self.memory[i].setdefault(j, deque([], maxlen=capacity))
        self.visit = {}
        for i in range(self.nState):
            self.visit.setdefault(i, {})
            for j in range(self.nAction):
                self.visit[i].setdefault(j, 0)

    def put_sample(self, transition):
        state_int = np.argmax(transition[0], axis=0)
        action = transition[1]
        self.visit[state_int][action] += 1
        self.total_visit += 1
        # if self.total_visit % 2000 == 0:
        #     for row in range(6):
        #         print([i[0] + i[1] + i[2] + i[3] for i in self.visit.values()][row * 10 : (row + 1) * 10])
        #     print()
        self.memory[state_int][action].append(transition)
        temp_list = []
        for i in range(self.nState):
            for j in range(self.nAction):
                temp_list += list(self.memory[i][j])
        self.memory_merge = deque(temp_list)

    def get_samples(self):
        """
        replay memory에서 batchsize 만큼의 (s,a,r,s',done_mask,bootstrap_mask)을 return
        """
        batch_size = min([len(self.memory_merge), self.batch_size])
        mini_batch = random.sample(self.memory_merge, batch_size)

        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_lst.append([done_mask])

        return torch.tensor(np.array(s_lst), dtype=torch.float), torch.tensor(np.array(a_lst)), torch.tensor(np.array(r_lst)), torch.tensor(np.array(s_prime_lst), dtype=torch.float), torch.tensor(np.array(done_lst))


#####################################################
# 모델 저장 시 실행
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from lava_grid import ZigZag6x10


def train_agent(agent, env):
    """
    agent class가 호출되면 자동으로 실행되는 함수
    agent를 학습시키고, 학습 시 sample efficiency(AUC)를 return
    """

    num_episode = agent.training_episode  # agent 학습시 사용할 episode 수
    cum_reward_list = list()
    reach_goal = 0

    for episode in tqdm(range(num_episode)):
        s = env.reset()
        s = np.eye(env.observation_space.n)[s]
        done = False
        cum_reward = 0.0
        episode_state_visit = {}
        for i in range(env.observation_space.n):
            episode_state_visit.setdefault(i, 0)

        while not done:
            # epsilon-greedy
            coin = random.random()
            if coin < agent.epsilon:
                action = random.randint(0, env.action_space.n - 1)

            else:
                action = agent.action(s)

            ns, reward, done, _ = env.step(action)

            # reward compensation
            state_int = np.argmax(s, axis=0)
            raw_reward = reward
            if raw_reward != -1:
                reward = reward + max(agent.alpha / (agent.replay_memory.visit[state_int][action] + 1) - 2, 0) / (agent.alpha / 4)
                reward = reward + max(-episode_state_visit[state_int] / 10, -0.5)
            # if episode % 100 == 0:
            #     print("state:", np.argmax(s), " action:", action, " reward:", reward)

            episode_state_visit[state_int] += 1

            done_mask = 0.0 if done else 1.0

            transition = (s, action, reward, ns, done_mask)

            agent.replay_memory.put_sample(transition)

            if episode > 25:
                agent.update()

            s = ns
            cum_reward += raw_reward

        if done and raw_reward == 1:
            reach_goal += 1

        if (episode + 1) % 100 == 0:
            print("success num and iter : ", reach_goal, episode + 1)

        if reach_goal <= 20:
            agent.epsilon = max(agent.epsilon - 1 / 500, 0.1)
        else:
            agent.epsilon = 0

        if episode % agent.target_update_frequency == 0:
            agent.update_target_network()

        cum_reward_list.append(cum_reward)
    AUC = np.sum(cum_reward_list)
    return AUC


if __name__ == "__main__":

    # default setting
    max_steps = 100
    stochasticity = 0  # probability of the selected action failing and instead executing any of the remaining 3
    no_render = True

    lava_env = ZigZag6x10(max_steps=max_steps, act_fail_prob=stochasticity, goal=(5, 9), numpy_state=False)

    batch_size = 512
    capacity = 5000
    epsilon = 0.2
    alpha = 100
    training_episode = 3000

    agent = agent(nState=lava_env.nS, batch_size=batch_size, capacity=capacity, epsilon=epsilon, alpha=alpha, training_episode=training_episode)
    # agent.load_weights()
    AUC = train_agent(agent=agent, env=lava_env)
    print(AUC)
    # agent.save_model()
