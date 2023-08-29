import numpy as np
import gym
import torch


from matplotlib import animation
import matplotlib.pyplot as plt

import replay_buffer_bounded
import actor
import critic

#from custompendulumenv import CustomPendulumEnv
#import laserhockey.hockey_env as h_env

class SAC():
    def __init__(self, env, gamma, tau, alpha, lr, buffer_maxlen,batchsize = 64,num_episodes = 80, max_steps = 1000, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), hidden_dim=256,
        double_hidden_layer=False, use_softplus_policy=False, lr_alpha=-1, HOCKEY_MODE=True):
        self.device = device
        self.device_cpu = torch.device("cpu")
        self.use_softplus_policy = use_softplus_policy
        self.double_hidden_layer = double_hidden_layer
        # CONFIGURATIONS
    
        self.hidden_dim = hidden_dim
        self.num_episodes= num_episodes
        self.max_steps = max_steps
        self.updates = 1
        self.update_rate = 50
        self.discount_factor = gamma
        self.soft_tau = tau
        self.batch_size = batchsize
        self.lr = lr
        
        self.HOCKEY_MODE = HOCKEY_MODE
        self.delay_step = 2
        #INITIALIZATIONS
        # if(self.HOCKEY_MODE):
        #     self.env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_SHOOTING)
        # else:
        #     self.env = gym.make(self.env_name)
        self.env = env
        if self.HOCKEY_MODE:
            self.action_dim = int(env.action_space.shape[0] / 2)
        else:
            self.action_dim = self.env.action_space.shape[0]
        self.state_dim  = np.array(self.env.reset()).shape[0]

        self.D = replay_buffer_bounded.Buffer(1,self.state_dim,self.action_dim,buffer_maxlen,self.device)

        self.episode_rewards =[]

        #INIT ACTOR

        self.policy_nw = actor.PolicyNetwork(self.state_dim, self.action_dim, self.hidden_dim, self.device, double_hidden_layer, use_softplus=use_softplus_policy)
        self.policy_optimizer =torch.optim.Adam(self.policy_nw.parameters(),lr=self.lr)

        if self.device.type == 'cpu':
            self.policy_nw_cpu = self.policy_nw
        else:
            self.policy_nw_cpu = actor.PolicyNetwork(self.state_dim, self.action_dim, self.hidden_dim, self.device_cpu, double_hidden_layer, use_softplus=use_softplus_policy)

            # copy params to policy_cpu
            for target_param, param in zip(self.policy_nw_cpu.parameters(), self.policy_nw.parameters()):
                target_param.data.copy_(param.to(device=self.device_cpu))

        #INIT CRITIC
        #Q1
        self.soft_q_nw1 = critic.SoftQNetwork(self.state_dim, self.action_dim, self.hidden_dim, self.device, double_hidden_layer)
        self.soft_q_loss1 = torch.nn.MSELoss()
        self.soft_q_opt1 = torch.optim.Adam(self.soft_q_nw1.parameters(), lr=self.lr*2)
        #Q2
        self.soft_q_nw2 = critic.SoftQNetwork(self.state_dim, self.action_dim, self.hidden_dim, self.device, double_hidden_layer)
        self.soft_q_loss2 = torch.nn.MSELoss()
        self.soft_q_opt2 = torch.optim.Adam(self.soft_q_nw2.parameters(), lr=self.lr*2)
        #Targets
        self.target_soft_q_nw1 = critic.SoftQNetwork(self.state_dim, self.action_dim, self.hidden_dim, self.device, double_hidden_layer)
        self.target_soft_q_nw2 = critic.SoftQNetwork(self.state_dim, self.action_dim, self.hidden_dim, self.device, double_hidden_layer)
        #copy params
        for target_param,param in zip(self.target_soft_q_nw1.parameters(), self.soft_q_nw1.parameters()):
            target_param.data.copy_(param.data)
        
        for target_param,param in zip(self.target_soft_q_nw2.parameters(), self.soft_q_nw2.parameters()):
            target_param.data.copy_(param.data)

        
        for param in self.target_soft_q_nw1.parameters():
            param.requires_grad = False

        for param in self.target_soft_q_nw2.parameters():
            param.requires_grad = False
        
        #INIT ENTROPY

        self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape)).to(self.device)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)  #since we look at entropy,update must operate in log space
        self.alpha = torch.exp(self.log_alpha)
        if (lr_alpha > 0):
            self.lr_alpha = lr_alpha
        else:
            self.lr_alpha = lr
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.lr_alpha)
        #self.alpha_optim = torch.optim.Adam([self.alpha], lr=self.lr)
        
        #RUN
        #self.run()

    def get_action_cpu(self, state):
        action = self.policy_nw_cpu.pred(torch.from_numpy(np.asarray(state)).float())
        action = action.detach().numpy()

        return action

    def sync_cpu_net(self):
        if not(self.device.type == 'cpu'):
            for target_param, param in zip(self.policy_nw_cpu.parameters(), self.policy_nw.parameters()):
                target_param.data.copy_(param.to(device=self.device_cpu))

    def update(self, update_steps, batch_size):
        #get batch
        state,action,reward, next_state,done = self.D.get_Sample_batch(batch_size)
        
        
        #GET QLOSS
        with torch.no_grad():
            new_action, log_prob, _ = self.policy_nw.sample_other(next_state)
    
            pred_target_q_1 = self.target_soft_q_nw1(next_state, new_action)
            pred_target_q_2 = self.target_soft_q_nw2(next_state, new_action)
            pred_target_q = torch.min(pred_target_q_1,pred_target_q_2)

            target_q_func = reward + self.discount_factor *(1-done)*(pred_target_q-self.alpha*log_prob)
            
            
        
        self.pred_q_value1 = self.soft_q_nw1(state,action)
        self.pred_q_value2 = self.soft_q_nw2(state,action)

        q_value_loss1 = self.soft_q_loss1(self.pred_q_value1, target_q_func.detach())
    
        q_value_loss2 = self.soft_q_loss2(self.pred_q_value2, target_q_func.detach())

        self.soft_q_opt1.zero_grad()
        q_value_loss1.backward()
        self.soft_q_opt1.step()

        self.soft_q_opt2.zero_grad()
        q_value_loss2.backward()
        self.soft_q_opt2.step()
        #print(q_value_loss2)
        #print(q_value_loss1)
        

        #GET POLICY LOSS
        new_action, log_prob ,_= self.policy_nw.sample_other(state)
        if update_steps % self.delay_step == 0:
            
            pred_q_value1 = self.soft_q_nw1(state,new_action)
            pred_q_value2 = self.soft_q_nw2(state,new_action)
        
            pred_q_value = torch.min(pred_q_value1,pred_q_value2)
            
            
            pol_loss = torch.mean(self.alpha*log_prob-pred_q_value)  
            #print(pol_loss)
            #print(self.alpha)
            #UPDATE NETWORKS

            

            self.policy_optimizer.zero_grad()
            pol_loss.backward()
            self.policy_optimizer.step()

                 #UPDATE TARGET NETWORKS

            for target_param, param in zip(self.target_soft_q_nw1.parameters(), self.soft_q_nw1.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau
                    
                )
            
        
            for target_param, param in zip(self.target_soft_q_nw2.parameters(), self.soft_q_nw2.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau
                    
                )


        #UPDATE ALPHA

        
        alpha_loss = torch.mean(-self.log_alpha * (self.target_entropy + log_prob).detach()) #
        
        
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self.alpha = self.log_alpha.exp()

    
    def run(self):
        
        frames = 0
        update_steps = 0
        for i in range(self.num_episodes):
            state = self.env.reset()
            episode_reward = 0
            print(f"episode: {i}, frame: {frames}, update_steps: {update_steps}")
            step = 0
            done = 0
            while (not bool(done)):
            #for step in range(self.max_steps):
                frames = frames + 1
                if(i <10): #gather data in start phase
                    action = self.env.action_space.sample()
                    
                    if(self.HOCKEY_MODE):
                        enemy_action = np.random.uniform(-1,1,4)
                        next_state,reward, done, _ = self.env.step(np.hstack([action,enemy_action]))#
                        obs_agent2 = self.env.obs_agent_two()
                        
                    else:    
                        next_state,reward, done, _ = self.env.step(action)
                    
                else:
                    action = self.policy_nw.pred(torch.from_numpy(np.asarray(state)).float())
                    action = action.detach().numpy()
                    if(self.HOCKEY_MODE):
                        enemy_action = np.random.uniform(-1,1,4)
                        next_state,reward, done, _ = self.env.step(np.hstack([action,enemy_action]))
                        obs_agent2 = self.env.obs_agent_two()
                        #print(obs_agent2)
                    else:    
                        next_state,reward, done, _ = self.env.step(action)
                next_state = np.squeeze(next_state)
                self.D.push(state,action,reward,next_state,done)
                if(bool(done)):
                    self.env.reset()
                    break
                state = next_state
                episode_reward+=reward
                
                if frames > self.batch_size:
                    for j in range (self.updates):
                        #print(step)
                        self.update(update_steps)        
                        update_steps += 1
                step+=1
            self.episode_rewards.append(episode_reward)
            print(f"reward : {episode_reward}")

    def save_frames_as_gif(self,frames, path='./', filename='gym_animation.gif'):
        #this function is taken from: https://gist.github.com/botforge/64cbb71780e6208172bbf03cd9293553
        #Mess with this to change frame size
        plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

        patch = plt.imshow(frames[0])
        plt.axis('off')

        def animate(i):
            patch.set_data(frames[i])

        anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=1, repeat = True)
        anim.save(path + filename, writer='pillow', fps=100)



    def render(self,e, save_gif = False):
        img_frames = []
        state = self.env.reset()
        for i in range(200):
            action = self.policy_nw.pred(torch.from_numpy(np.asarray(state)).float())
            action = action.detach().numpy()
            if(self.HOCKEY_MODE):
                enemy_action = np.random.uniform(-1,1,4)
                next_state,reward, done, _ = self.env.step(np.hstack([action,enemy_action]))
            else:    
                next_state,reward, done, _ = self.env.step(action)
            if(done):
                state = self.env.reset()
            state = np.squeeze(next_state)
            img_frames.append(self.env.render(mode="rgb_array"))
        self.env.close()
        if save_gif:
            self.save_frames_as_gif(img_frames,path = 'imgs/',filename = f'pendulum_e_{e}.gif')

    