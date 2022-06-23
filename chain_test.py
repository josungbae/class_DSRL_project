from tkinter import S
from chain_mdp import ChainMDP
from agent_chainMDP import agent


# receive 1 at rightmost state and receive small reward at leftmost state
env = ChainMDP(10)


""" Your agent """
agent = agent(env, training=True) # training=True이면 여기서 training을 진행하고, training이 완료된 agent를 호출.
                                  # training=False이면 저장된 모델을 불러옴.
print('AUC:',agent.AUC)
s = env.reset()
done = False
cum_reward = 0.0

while not done:    
    action = agent.action(s)
    ns, reward, done, _ = env.step(action)
    print('\nstate:',s)
    print('action:',action)
    print('done:',done)

    cum_reward += reward
    s = ns # state transition

print(f"total reward: {cum_reward}")
