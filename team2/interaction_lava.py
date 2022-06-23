import numpy as np
import random


def calculate_performance(episodes, env, agent):

    episodic_returns = []

    for epi in range(episodes):

        s = env.reset()
        s = np.eye(env.observation_space.n)[s]

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
    reach_goal = 0

    for episode in range(episodes):

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
            print("success num, iter and cum_reward : ", reach_goal, episode + 1, cum_reward)

        if reach_goal <= 20:
            agent.epsilon = max(agent.epsilon - 1 / 500, 0.1)
        else:
            agent.epsilon = 0

        if episode % agent.target_update_frequency == 0:
            agent.update_target_network()

            #####################
            # If your agent needs to update the weights at every time step, complete your update process in this area.
            # e.g., agent.update()

            #####################
        #####################
        # elif your agent needs to update the weights at the end of every episode, complete your update process in this area.
        # e.g., agent.update()

        #####################

        episodic_returns.append(cum_reward)
    print(cum_reward)
    return np.sum(episodic_returns)

