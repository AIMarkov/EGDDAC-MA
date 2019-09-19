import gym
import pybullet_envs
from config import load_config
from modules import DDPG, OrnsteinUhlenbeckNoise, to_scalar
import numpy as np
from itertools import count
import csv



cf = load_config('config/baseline.py')
env = gym.make('Walker2DBulletEnv-v0') #Humanoid
out1 = open('Walker2DBulletEnv_a.csv', 'a', newline='')
out2 =open("e_Walker2DBulletEnv_a.csv",'a',newline='')
csv_writer = csv.writer(out1, dialect='excel')
ex_writer=csv.writer(out2,dialect='excel')
cf.state_dim = env.observation_space.shape[0]
cf.action_dim = env.action_space.shape[0]
cf.scale = float(env.action_space.high[0])

print(' State Dimensions: ', cf.state_dim)
print(' Action Dimensions: ', cf.action_dim)
print('Action low: ', env.action_space.low)
print('Action high: ', env.action_space.high)

noise_process = OrnsteinUhlenbeckNoise(cf)
model = DDPG(cf)
for i in range(cf.num_agent):
    model.copy_weights(model.actor[i], model.actor_target[i])
model.copy_weights(model.critic, model.critic_target)

losses = []
total_timesteps = 0

print('num_agent:',cf.num_agent)
print('cf.stage:',cf.stage)
for epi in range(cf.max_episodes):
    s_t = env.reset()
    avg_reward = 0
    trace=[]
    for t in range(1000):
        T = int(t / cf.stage)
        o_t,a_t = model.sample_action(s_t,T)
        ex_at=model.sample_ex(o_t)
        a_t = a_t + (ex_at-a_t)*1e-5
        s_tp1, r_t, done, info = env.step(a_t)
        o_tp1=model.Encode_state(s_tp1,T)
        trace.append((o_t, a_t))
        if model.buffer[0].len <= cf.replay_start_size:
            for i in range(cf.num_agent):
                model.buffer[i].add(s_t, a_t, r_t,s_tp1,o_tp1 , float(done == False))
        else:
            model.buffer[T].add(s_t, a_t, r_t, s_tp1,o_tp1, float(done == False))
        avg_reward += r_t

        if done:
            break
        else:
            s_t = s_tp1

        if model.buffer[0].len >= cf.replay_start_size:
            _loss_a, _loss_c = model.train_batch(T)
            losses.append(to_scalar([_loss_a, _loss_c]))
    if epi<=10:
        model.Buffer.add_episode(trace,avg_reward)
    elif avg_reward>=model.Buffer.aver_R():
        model.Buffer.add_episode(trace,avg_reward)
    if len(losses) > 0:
        total_timesteps += t
        avg_loss_a, avg_loss_c = np.asarray(losses)[-100:].mean(0)
        print(
            'Episode {}: actor loss: {} critic loss: {}\
            episode_reward: {} episode_num_step: {} tot_timesteps: {}'.format(
             epi, avg_loss_a, avg_loss_c, avg_reward, t, total_timesteps
            ))
        csv_writer.writerow([epi, avg_loss_a, avg_loss_c, avg_reward, t, total_timesteps])

        s_t = env.reset()
        avg_reward = 0
        o_t = model.Encode_state(s_t, 0)
        for t in range(1000):
            T = int(t / cf.stage)
            g_at = model.sample_ex(o_t)
            s_tp1, r_t, done, info = env.step(g_at)
            o_tp1 = model.Encode_state(s_tp1, T)
            avg_reward += r_t
            if done:
                break
            else:
                o_t = o_tp1
        ex_writer.writerow([epi, avg_reward, t, total_timesteps])

    if (epi + 1) % 100 == 0:
        model.save_models()
print('Completed training!')
