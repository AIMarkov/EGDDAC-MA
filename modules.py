import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random


def to_var(arr, requires_grad=True, volatile=False):
    return Variable(torch.from_numpy(
        np.asarray(arr).astype('float32')
    ).cuda(), requires_grad=requires_grad, volatile=volatile)  #数据转换成variable


def to_scalar(arr):
    return [x.cpu().data.tolist() for x in arr]  #


class ReplayBuffer:  #经验池
    def __init__(self, cf):
        self.buffer_size = cf.max_buffer
        self.len = 0

        self.buffer = deque(maxlen=self.buffer_size)

    def sample(self, count):
        count = min(count, self.len)
        batch = random.sample(self.buffer, count)
        s_t, a_t, r_t, s_tp1,o_tp1, term = zip(*batch)
        a_t = to_var(a_t)
        r_t = to_var(r_t)
        s_t = to_var(s_t)
        term = to_var(term)
        s_tp1 = to_var(s_tp1)
        o_tp1=to_var(o_tp1)
        return s_t, a_t, r_t, s_tp1,o_tp1, term

    def add(self, s_t, a_t, r_t, s_tp1,o_tp1, term):
        transition = (s_t, a_t, r_t, s_tp1,o_tp1, term)
        self.len += 1
        if self.len > self.buffer_size:
            self.len = self.buffer_size
        self.buffer.append(transition)
class ReplayBuffer1:  #经验池
    def __init__(self, cf):
        self.buffer_size = cf.max_buffer1#2000
        self.len = 0
        self.R_list=deque(maxlen=50)#累积奖赏的平均
        self.buffer = deque(maxlen=self.buffer_size)

    def sample(self, count):
        count = min(count, self.len)
        batch = random.sample(self.buffer, count)
        s_t, a_t = zip(*batch)
        a_t = to_var(a_t)
        s_t = to_var(s_t)
        return s_t, a_t

    def add_episode(self,trace,average_R):
        self.buffer.extend(trace)
        self.len+=len(trace)
        self.R_list.append(average_R)
    def aver_R(self):
        return sum(self.R_list)/len(self.R_list)


class OrnsteinUhlenbeckNoise():#噪音
    def __init__(self, cf):
        self.action_dim = cf.action_dim

        self.mu_val = cf.mu
        self.sigma_val = cf.sigma
        self.theta = cf.theta
        self.dt = cf.dt

    def reset(self):
        self.mu = np.zeros(self.action_dim, dtype='float32') + self.mu_val
        self.sigma = np.ones(self.action_dim, dtype='float32') * self.sigma_val
        self.X = np.zeros_like(self.mu)

    def sample(self):
        epsilon = np.random.normal(size=self.mu.shape).astype('float32')
        term1 = self.theta * (self.mu - self.X) * self.dt
        term2 = self.sigma * np.sqrt(self.dt) * epsilon
        self.X += term1 + term2
        return self.X


class LayerNorm(nn.Module):  #做一个正则化处理
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features)) #nn.Parameter也是一种Variable
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
class Encode(nn.Module):
    def __init__(self,cf):
        super(Encode,self).__init__()
        self.model=nn.Sequential(
            nn.Linear(cf.state_dim,300),
            LayerNorm(300),
            nn.ReLU(),
            nn.Linear(300,100),
            LayerNorm(100),
            nn.Tanh(),
            nn.Linear(100,cf.signal_dim),
            nn.Tanh()
        )
        for i in [0, 3, 6]:
            nn.init.xavier_uniform(self.model[i].weight.data)
    def forward(self, input):
        return self.model(input)


class Actor(nn.Module):
    def __init__(self, cf,encoder):
        super(Actor, self).__init__()
        self.scale = cf.scale
        self.encode=encoder
        self.model = nn.Sequential(
            nn.Linear(cf.signal_dim, 200),#cf.state是状态
            LayerNorm(200),
            nn.ReLU(),
            nn.Linear(200, 100),
            LayerNorm(100),
            nn.ReLU(),
            nn.Linear(100, cf.action_dim),#cf.action是动作
            nn.Tanh()
        )

        for i in [0, 3]:
            nn.init.xavier_uniform(self.model[i].weight.data)
        self.model[-2].weight.data.uniform_(-3e-3, 3e-3)
    def forward(self, state,sample=None):
        x = self.encode(state)
        if sample is None:
            return self.model(x)*self.scale
        else:
            return x,self.model(x) * self.scale


class ex_actor(nn.Module):
    def __init__(self, cf):
        super(ex_actor, self).__init__()
        self.scale = cf.scale
        self.model = nn.Sequential(
            nn.Linear(cf.signal_dim, 200),#cf.state是状态,200
            LayerNorm(200),#200
            nn.ReLU(),
            nn.Linear(200, 100),#400,300
            LayerNorm(100),#300
            nn.ReLU(),
            nn.Linear(100, cf.action_dim),#cf.action是动作300
            nn.Tanh()
        )
        for i in [0, 3]:
            nn.init.xavier_uniform(self.model[i].weight.data)

        self.model[-2].weight.data.uniform_(-3e-3, 3e-3)
    def forward(self, signal):
        return self.model(signal) * self.scale


class Critic(nn.Module):#this  alse can use encode
    def __init__(self, cf):
        super(Critic, self).__init__()
        self.transform_state = nn.Sequential(
            nn.Linear(cf.state_dim, 400),
            LayerNorm(400),
            nn.ReLU()
            )
        nn.init.xavier_uniform(self.transform_state[0].weight.data)

        self.transform_both = nn.Sequential(
            nn.Linear(400 + cf.action_dim, 300),
            LayerNorm(300),
            nn.ReLU(),
            nn.Linear(300, 1)
            )
        nn.init.xavier_uniform(self.transform_both[0].weight.data)
        self.transform_both[-1].weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, o, action):
        state = self.transform_state(o)
        both = torch.cat([state, action], 1)
        return self.transform_both(both)


class DDPG(nn.Module):
    def __init__(self, cf):
        super(DDPG, self).__init__()
        self.cf = cf
        self.encoder=[Encode(cf).cuda() for i in range(cf.num_agent)]
        self.encoder_target=[Encode(cf).cuda() for i in range(cf.num_agent)]
        self.actor=[Actor(cf,self.encoder[i]).cuda() for i in range(cf.num_agent)]
        self.actor_target=[Actor(cf,self.encoder_target[i]).cuda() for i in range(cf.num_agent)]
        self.actor_optimizer=[optim.Adam(self.actor[i].parameters(),lr=cf.actor_learning_rate[i]) for i in range(cf.num_agent)]

        self.ex_actor=ex_actor(cf).cuda()
        self.ex_actor_optimizer=optim.Adam(self.ex_actor.parameters(),lr=cf.ex_learning_rate)

        self.critic = Critic(cf).cuda()
        self.critic_target = Critic(cf).cuda()
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=cf.critic_learning_rate
        )

        self.buffer=[ReplayBuffer(cf) for i in range(cf.num_agent)]

        self.Buffer=ReplayBuffer1(cf) #好的经验网

    def update_targets(self, model, target):
        for p, target_p in zip(model.parameters(), target.parameters()):
            target_p.data.copy_(
                self.cf.tau * p.data + (1-self.cf.tau) * target_p.data
            )

    def copy_weights(self, model, target):
        for p, target_p in zip(model.parameters(), target.parameters()):
            target_p.data.copy_(
                p.data
            )
    def Encode_state(self,state,T):
        state=to_var(state,volatile=True,requires_grad=False)
        result=self.encoder[T](state)
        return result.cpu().data.numpy()
    def sample_action(self, state,T):
        state = to_var(state, volatile=True, requires_grad=False)  #???????
        result = self.actor[T](state[None], 1)
        signal = result[0][0].cpu().data.numpy()  # ????
        action = result[1][0].cpu().data.numpy()
        return signal,action
    def sample_ex(self,state):
        state = to_var(state, volatile=True, requires_grad=False)  # ???????
        action = self.ex_actor(state[None])[0].cpu().data.numpy()  # ????
        return action


    def train_batch(self,T):

        for i in [T]:
            s_t, a_t, r_t, s_tp1,o_tp1, term = self.buffer[i].sample(self.cf.batch_size)

            a_tp1 = self.actor_target[i](s_tp1)
            q_value = self.critic_target(s_tp1, a_tp1).squeeze()

            ex_tp1 = self.ex_actor(o_tp1)
            ex_qvalue = self.critic_target(s_tp1, ex_tp1).squeeze()

            q_value = q_value + (ex_qvalue - q_value) * 1e-5



            td_target= r_t + self.cf.gamma * term * q_value
            td_current = self.critic(s_t, a_t).squeeze()
            delta=(td_target-td_current)
            td_target=delta*(F.tanh(r_t)+1)/2+td_current
            critic_loss = F.smooth_l1_loss(td_current, td_target.detach())

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            #actor
            self.actor_optimizer[i].param_groups[0]['lr'] = self.cf.actor_learning_rate[T]
            a_t_pred = self.actor[i](s_t)#也用旧的
            q_pred = self.critic(s_t, a_t_pred)
            actor_loss = -1 * q_pred.squeeze(1).mean()

            self.actor_optimizer[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizer[i].step()



            self.update_targets(self.actor[i], self.actor_target[i])
            self.update_targets(self.critic, self.critic_target)

        o_ex, a_ex = self.Buffer.sample(self.cf.batch_size)
        ex_a=self.ex_actor(o_ex)
        ex_actorloss=F.smooth_l1_loss(ex_a, a_ex.detach())

        self.ex_actor_optimizer.zero_grad()
        ex_actorloss.backward()
        self.ex_actor_optimizer.step()
        return actor_loss, critic_loss

    def save_models(self):
        for i in range(self.cf.num_agent):
            torch.save(self.actor[i].state_dict(), 'models/best_actor'+str(i)+'.model')
            torch.save(self.actor_target[i].state_dict(), 'models/best_actor_target'+str(i)+'.model')
        torch.save(self.critic.state_dict(), 'models/best_critic.model')
        torch.save(self.critic_target.state_dict(), 'models/best_critic_target.model')
        torch.save(self.ex_actor.state_dict(), 'models/ex_actor.model')




    def load_models(self):
        for i in range(self.cf.num_agent):
            self.actor[i].load_state_dict(torch.load('models/best_actor'+str(i)+'.model'))
        self.critic.load_state_dict(
            torch.load('models/best_critic.model'))
    def load_G_and_D(self):
        self.G.load_state_dict(
            torch.load('models/best_G_D_InvertedDoublePendulum/best_G.model')
        )
        self.D.load_state_dict(
            torch.load('models/best_G_D_InvertedDoublePendulum/best_D.model')
        )
