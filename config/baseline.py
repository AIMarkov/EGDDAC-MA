# rl parameters
import math
max_episodes = 80000
max_steps = 1000
max_buffer = 300000
max_buffer1=100000 #存储情节，情节个数
replay_start_size = 5000
signal_dim=20
stage=10
num_agent=int(1000/stage)+1



# noise parameters
mu = 0
theta = 0.15
sigma = 0.2
dt = 1e-2


#actor_learning_rate =[1e-4 for i in range(num_agent)]
actor_learning_rate =[1e-5 for i in range(6)]
actor_learning_rate.extend([1e-5 for i in range(num_agent-6)])#根据周期来设置学习率，前面的学习率可以小一点
ex_learning_rate=0.0002# 1e-5  #
critic_learning_rate = 1e-3
batch_size = 64

gamma = 0.99
tau = 0.001

