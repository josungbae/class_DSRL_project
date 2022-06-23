import numpy as np

def calculate_performance(episodes, env, agent):

    episodic_returns = []

    for epi in range(episodes):
        
        # agent.k = np.random.randint(agent.num_k) # test시 앙상블이 허용되지 않을 경우 이 코드 사용
        s = env.reset()

        done = False
        cum_reward = 0.0

        while not done:    
            action = agent.action(s)
            ns, reward, done, _ = env.step(action)
            cum_reward += reward
            s = ns
        
        episodic_returns.append(cum_reward)
                    
    return np.sum(episodic_returns)

def calculate_sample_efficiency(episodes, env, agent):

    episodic_returns = []
    
    for epi in range(episodes):
        
        s = env.reset()

        done = False
        cum_reward = 0.0
        """
        bootstrapped DQN에서 episode 생성을 위한 network 번호를 선택
        """
        agent.k = np.random.randint(agent.num_k)

        while not done:    
            action = agent.action(s)
            ns, reward, done, _ = env.step(action)
            cum_reward += reward
            # s = ns
            
            #####################
            # If your agent needs to update the weights at every time step, complete your update process in this area.
            # e.g., agent.update()
            done_mask = 0.0 if done else 1.0
            bootstrap_mask = np.random.binomial(1, 0.5, agent.num_k)
            """
            done_mask: episode 종료면 0, 아니면 1
            bootstrap_mask: 해당 transition data를 각 Q network 업데이트에 사용할지 안할지 여부
                ex) bootstrap_mask가 [1,0,0,1,1]이면, 해당 transition 데이터는 1,4,5번 network 업데이트에만 사용됨 
            """
            transition = (s, action, reward, ns, done_mask, bootstrap_mask)

            agent.replay_memory.put_sample(transition)
            
            # if agent.replay_memory.len() > agent.batch_size:
            if epi > 10:
                agent.update()
            #####################
            """
            s=ns를 replay memory 저장을 위해 아래로 이동하였습니다
            """
            s = ns 
        #####################
        # elif your agent needs to update the weights at the end of every episode, complete your update process in this area.
        # e.g., agent.update()
        if epi % agent.target_update_frequency == 0:
            agent.update_target_network()

        #####################
        
        episodic_returns.append(cum_reward)
                    
    return np.sum(episodic_returns)

