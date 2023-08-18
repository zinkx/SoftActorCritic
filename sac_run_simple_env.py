import gym
import torch
import numpy as np
import math
import random
from torch.utils.tensorboard import SummaryWriter
#from heatmap import HeatMap
from SAC import SAC


env = gym.make("Pendulum-v0") # try these 'MountainCarContinuous-v0' 'LunarLanderContinuous-v2'

# SAC Params
gamma = 0.99
tau = 0.01
alpha = 0.2
lr = 3e-4
batch_size = 64
episodes = 550
buffer_maxlen = 1000000
max_episodes = 5000
state = env.reset()

agent = SAC(env, gamma, tau, alpha, lr, buffer_maxlen, device= torch.device('cpu'),hidden_dim=128, HOCKEY_MODE = False)

def train(env, agent, max_episodes, batch_size):
    frames = 0
    update_steps = 0
    for i in range(episodes):
        if i % 100 == 99:
            agent.render()
        agent.sync_cpu_net()
        state = env.reset()
        episode_reward = 0
        print(f"episode: {i}, frame: {frames}, update_steps: {update_steps}")
        step = 0
        done = 0
        while (not bool(done)):
            frames = frames + 1
            if(i <10): #gather random data in start phase
                action = env.action_space.sample()
                next_state,reward, done, _ = env.step(action)
            else:
                action = agent.get_action_cpu(state)
                next_state,reward, done, _ = env.step(action)
            next_state = np.squeeze(next_state)
            agent.D.add_Batch(state,action,reward,next_state,done)
            if(bool(done)):
                env.reset()
                break
            state = next_state
            episode_reward+=reward
                
            #if(step%self.update_rate==0):
            if frames > batch_size:
                for j in range (agent.updates):
                    agent.update(update_steps,batch_size)        
                    update_steps += 1
            step+=1
        agent.episode_rewards.append(episode_reward)
        print(f"reward : {episode_reward}")

#agent.run()
train(env, agent,max_episodes, batch_size)