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

PATH = './team2_lava_model_controller.pth'
METAPATH = './team2_lava_model_metacontroller.pth'

class agent():
    
    def __init__(self, env, training=True):

        # self.sample_actions = [3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
        
        self.env = env
        self.input_size = self.env.observation_space.n
        self.capacity = 20000 # replay memory 크기
        self.meta_capacity = 200
        self.batch_size = 100
        self.meta_batch_size = 10 
        self.meta_epsilon = 1.0
        self.epsilon = 1.0 # epsilon-greedy
        self.training_episode = 3000
        self.gamma = 1.0
        self.learning_rate = 0.01
        self.meta_learning_rate = 0.01
        self.target_update_frequency = 1 # episode 단위
        self.meta_target_update_frequency = 4
        # self.num_goal = self.env.observation_space.n # 모든 state를 goal 후보로 지정
        self.num_goal = 5 # (4,1), (4,4), (1,5), (1,8), (5,9)를 subgoal 후보로 지정
        self.num_sa = np.ones((self.env.observation_space.n,self.num_goal))

        self.meta_policy_network = MetaController(self.input_size, self.num_goal)
        self.meta_target_network = MetaController(self.input_size, self.num_goal)
        self.policy_network = Controller(self.input_size + self.num_goal, self.env.action_space.n)
        self.target_network = Controller(self.input_size + self.num_goal, self.env.action_space.n)

        self.meta_replay_memory = ReplayMemory(self.meta_capacity, self.meta_batch_size)
        self.replay_memory = ReplayMemory(self.capacity, self.batch_size)

        self.meta_optimizer = optim.Adam(self.meta_policy_network.parameters(), lr=self.meta_learning_rate)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)

        if training:
            self.AUC, self.goal_count, self.cum_reward_list, self.controller_success = self.train_agent()
            self.save_model(PATH, METAPATH)
        else:
            self.policy_network.load_state_dict(torch.load(PATH))
            self.meta_policy_network.load_state_dict(torch.load(METAPATH))

    def save_model(self, path, meta_path):
        torch.save(self.policy_network.state_dict(), path)
        torch.save(self.meta_policy_network.state_dict(), meta_path)

    def action(self, state_and_goal):
        output = self.policy_network(state_and_goal)
        action = int(torch.argmax(output))

        return action

    def exp_bonus(self,state,goal):
        return np.sqrt(2*np.log(self.num_sa.sum())/self.num_sa[np.where(state==True)[0][0],goal])
    
    def select_goal(self, state):
        output = self.meta_policy_network(state)
        bs = torch.tensor([self.exp_bonus(state,a) for a in range(self.num_goal)])
        
        goal = int(torch.argmax(output + bs))
        # print('goal value:',output)
        # goal = int(torch.argmax(output))

        return goal

    def update(self): # controller 업데이트

        s, a, r, ns, done_mask = self.replay_memory.get_samples()

        q_out = self.policy_network(s)
        q_a = q_out.gather(1,a)

        max_q_prime = self.target_network(ns).max(1)[0].unsqueeze(1)
        target = r + self.gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def meta_update(self): # meta controller 업데이트
    
        s, a, r, ns, done_mask = self.meta_replay_memory.get_samples()

        q_out = self.meta_policy_network(s)
        q_a = q_out.gather(1,a)

        max_q_prime = self.meta_target_network(ns).max(1)[0].unsqueeze(1)
        target = r + self.gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)

        self.meta_optimizer.zero_grad()
        loss.backward()
        self.meta_optimizer.step()
    
    def update_target_network(self):

        self.target_network.load_state_dict(self.policy_network.state_dict())

    def update_meta_target_network(self):
    
        self.meta_target_network.load_state_dict(self.meta_policy_network.state_dict())

    def train_agent(self):
        """
        agent class가 호출되면 자동으로 실행되는 함수
        agent를 학습시키고, 학습 시 sample efficiency(AUC)를 return
        """
        num_episode = self.training_episode # agent 학습시 사용할 episode 수
        cum_reward_list = list()
        goal_num = 0 # meta-controller가 controller에게 goal을 제시한 횟수
        goal_count = 0 # controller의 최종 goal 도달 횟수

        controller_success = list()
        for episode in tqdm(range(num_episode)):
            s = self.env.reset()
            s = np.eye(self.env.observation_space.n)[s]
            done = False

            # # Epsilon-Greedy goal selection
            # coin = random.random()
            # if coin < self.meta_epsilon:
            #     goal = random.randint(0, self.num_goal-1)
            # else:
            #     goal = self.select_goal(s)
            goal = self.select_goal(s)
            # print('goal:',goal)
            onehot_goal = np.eye(self.num_goal)[goal] # controller의 input으로 들어가는 goal의 형태는 one-hot encoding된 형태
            # goal 도달 확인 및 출력용 index 할당
            if goal == 0:
                goal_index = (4,1)
            elif goal == 1:
                goal_index = (4,4)
            elif goal == 2:
                goal_index = (1,5)
            elif goal == 3:
                goal_index = (1,8)
            else:
                goal_index = (5,9)
            """
            [1,0,0,0,0]: (4,1), 
            [0,1,0,0,0]: (4,4), 
            [0,0,1,0,0]: (1,5),
            [0,0,0,1,0]: (1,8),
            [0,0,0,0,1]: (5,9)
            """
            goal_num += 1
            self.num_sa[np.where(s==True)[0][0],goal] += 1

            # 주어진 goal에 대해 controller가 episode 진행
            while not done:    
                total_extrinsic_reward = 0
                s_0 = s # s_0: 주어진 goal에 대해 episode를 진행할 때 controller의 initial state
                goal_reached = False # controller의 sub-goal 도달 여부

                """episode 시작"""
                while not done and not goal_reached:    
                    # Epsilon-Greedy action selection
                    coin = random.random()
                    if coin < self.epsilon:
                        action = random.randint(0, self.env.action_space.n-1)
                    else:
                        action = self.action(np.concatenate([s,onehot_goal]))

                    # Transition    
                    ns, extrinsic_reward, done, _ = self.env.step(action)
                    if extrinsic_reward > 0:
                        goal_count +=1

                    # controller가 sub-goal에 도달하면 1의 intrinsic reward 부여
                    # if np.unravel_index(np.where(ns==True)[0][0],(6,10)) == np.unravel_index(np.where(onehot_goal==True)[0][0],(6,10)):
                    if np.unravel_index(np.where(ns==True)[0][0],(6,10)) == goal_index:
                        intrinsic_reward = 1
                        goal_reached = True
                        controller_success.append(1)
                        print('\n\nepisode:',episode)
                        print('controller 성공 시 주어진 goal:',goal_index)
                        print('controller 성공 확률:', np.sum(controller_success) / goal_num)
                    else:
                        intrinsic_reward = 0
                        goal_reached = False
                        controller_success.append(0)
                    
                    # 매 transition data를 controller의 replay memory에 저장
                    transition = (np.concatenate([s,onehot_goal]), action, intrinsic_reward, np.concatenate([ns,onehot_goal]), goal_reached) # 여기 goal_reached 말고 done..?
                    self.replay_memory.put_sample(transition)

                    # update controller & meta-controller
                    if episode > 500:
                        self.meta_update()
                    if episode > 200:
                        self.update()
                    
                    total_extrinsic_reward += extrinsic_reward
                    s = ns
                    
                # controller가 sub-goal에 도달하거나 episode가 끝나면 meta-controller의 replay memory에 data 저장
                meta_transition = (s_0, goal, total_extrinsic_reward, ns, done)
                # print('meta transition:', meta_transition)
                self.meta_replay_memory.put_sample(meta_transition)

                # episode가 끝나지 않았을 경우 현재까지 온 state에서 새로운 goal 부여
                if not done:
                    # # Epsilon-Greedy goal selection
                    # coin = random.random()
                    # if coin < self.meta_epsilon:
                    #     goal = random.randint(0, self.num_goal-1)
                    # else:
                    #     goal = self.select_goal(s)
                    goal = self.select_goal(s)
                    onehot_goal = np.eye(self.num_goal)[goal]
                    if goal == 0:
                        goal_index = (4,1)
                    elif goal == 1:
                        goal_index = (4,4)
                    elif goal == 2:
                        goal_index = (1,5)
                    elif goal == 3:
                        goal_index = (1,8)
                    else:
                        goal_index = (5,9)
                    goal_num += 1
                    self.num_sa[np.where(s==True)[0][0],goal] += 1
                    print('goal 성공 후 새롭게 주어진 goal:',goal_index)
                """episode 종료"""

                # # decay the epsilon
                # if episode > 500:
                #     self.meta_epsilon = max(self.meta_epsilon-1/1500, 0.1)
                # if episode > 200:
                #     self.epsilon = max(self.epsilon-1/2000, 0.1)
                #     # if np.sum(controller_success) / goal_num > 0.3:
                #     #     self.epsilon = 0.1

                # # controller가 sub-goal에 도달하거나 episode가 끝나면 meta-controller의 replay memory에 data 저장
                # meta_transition = (s_0, goal, total_extrinsic_reward, ns, done)
                # print('meta transition:', meta_transition)
                # self.meta_replay_memory.put_sample(meta_transition)
            # decay the epsilon
            if episode > 500:
                self.meta_epsilon = max(self.meta_epsilon-1/1500, 0.1)
            if episode > 200:
                self.epsilon = max(self.epsilon-1/2000, 0.1)
            
            # target network update
            if episode % self.target_update_frequency == 0:
                self.update_target_network()
            if episode % self.meta_target_update_frequency == 0:
                self.update_meta_target_network()

            cum_reward_list.append(total_extrinsic_reward)
            AUC = np.sum(cum_reward_list)

        return AUC, goal_count, cum_reward_list, controller_success            


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

class MetaController(nn.Module):
    def __init__(self, input_size, output_size):
        """
        Initialize a Meta-Controller of Hierarchical DQN
            input: onehot-encoded state
            output: value of goals
        """
        super(MetaController, self).__init__()
        self.fc1 = nn.Linear(input_size, 120) # input_size = 60
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, output_size) # -> goal 후보 개수

    def forward(self, x):

        x = torch.tensor(x, dtype=torch.float32)
        # x = x.clone().detach()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.fc3(x)

class Controller(nn.Module):
    def __init__(self, input_size, output_size=4):
        """
        Initialize a Controller(given goal) of h-DQN
            input: onehot-encoded state + goal
            output: value of actions
        """
        super(Controller, self).__init__()
        self.fc1 = nn.Linear(input_size, 200)
        self.fc2 = nn.Linear(200, 50)
        self.fc3 = nn.Linear(50, output_size) # -> action 4개

    def forward(self, x):

        x = torch.tensor(x, dtype=torch.float32)
        # x = x.clone().detach()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.fc3(x)