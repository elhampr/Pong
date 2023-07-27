import os
import sys
import time
import argparse
import numpy as np
from collections import namedtuple 


import gym
import torch.nn as nn
import torch
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn.utils as nn_utils
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import ptan


ENV_NAME = "PongNoFrameskip-v4"
ENV_COUNT = 32
HIDDEN_LAYER = 512
LEARNING_RATE = 0.01
PROCESSES_COUNT = 4
GAMMA = 0.99
LEARNING_RATE = 0.001
REWARD_STEPS = 4
GRAD_BATCH_SIZE = 32
BATCH_SIZE = 128
ENTROPY_BETA = 0.01
CLIP_GRAD = 0.1


TotalReward = namedtuple('TotalReward', field_names='reward')

#to know more about kernel size and stride: https://d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html
#to know more about relu: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
class AtariPGN(nn.Module):
    def __init__(self, input_size, output_size):
        super(AtariPGN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_size, 32, kernel_size=8, stride=4),
            nn.ReLU(),  
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),  
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU() 
        )
        
        conv_out_size = self._get_conv_out(input_size)
        
        self.actor = nn.Sequential(
            nn.Linear(conv_out_size, HIDDEN_LAYER),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER, output_size)
        )

        self.critic = nn.Sequential(
            nn.Linear(conv_out_size, HIDDEN_LAYER),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER, 1)
        ) 

    #This is bc we do not know the output size from conv (for any input).
    #here we apply conv on a fake tensor with same input size and then find the output size
    def _get_conv_out(self, shape):
        fake = self.conv(torch.zeros(1, *shape))
        return int(np.prod(fake.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.actor(conv_out), self.critic(conv_out)
    
class RewardTracker:
    def __init__(self, writer, stop_reward):
        self.writer = writer
        self.stop_reward = stop_reward

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0
        self.total_rewards = []
        return self

    def __exit__(self, *args):
        self.writer.close()

    def reward(self, reward, frame, epsilon=None):
        self.total_rewards.append(reward)
        speed = (frame - self.ts_frame) / (time.time() - self.ts)
        self.ts_frame = frame
        self.ts = time.time()
        mean_reward = np.mean(self.total_rewards[-100:])
        epsilon_str = "" if epsilon is None else ", eps %.2f" % epsilon
        print("%d: done %d games, mean reward %.3f, speed %.2f f/s%s" % (
            frame, len(self.total_rewards), mean_reward, speed, epsilon_str
        ))
        sys.stdout.flush()
        if epsilon is not None:
            self.writer.add_scalar("epsilon", epsilon, frame)
        self.writer.add_scalar("speed", speed, frame)
        self.writer.add_scalar("reward_100", mean_reward, frame)
        self.writer.add_scalar("reward", reward, frame)
        if mean_reward > self.stop_reward:
            print("Solved in %d frames!" % frame)
            return True
        return False


def unpack(batch, net, device = "cpu"):
    
    states, actions, rewards, next_states, not_last = [], [], [], [], []

    for idx, exp in enumerate(batch):
        states.append(np.array(exp.state, copy=False))
        actions.append(int(exp.action))
        rewards.append(exp.reward)
        if exp.last_state is not None:
            next_states.append(np.array(exp.last_state, copy=False))
            not_last.append(idx)

    states_tf = torch.FloatTensor(states).to(device)       
    actions_tf = torch.LongTensor(actions).to(device)
    
    rewards_np = np.array(rewards, dtype=np.float32)
    
    if not_last:
        next_states_tf = torch.FloatTensor(next_states).to(device) 
        next_state_val_tf = net(next_states_tf)[1]
        next_state_val_np = next_state_val_tf.data.cpu().numpy()[:,0]
        rewards_np += GAMMA**REWARD_STEPS*next_state_val_np

    exp_reward_tf = torch.FloatTensor(rewards_np).to(device)

    return states_tf, actions_tf, exp_reward_tf


def grad_func(name, net, device, train_q):
    
    make_env = lambda: ptan.common.wrappers.wrap_dqn(gym.make(ENV_NAME))
    envs = [make_env() for _ in range(ENV_COUNT)]

    agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], device=device, apply_softmax=True)

    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, gamma=GAMMA, steps_count=REWARD_STEPS)

    trj_indx = 0
    batch = []

    with RewardTracker(writer, 18) as tracker: #checks the reward with threshold
        with ptan.common.utils.TBMeanTracker(writer, 100) as tb_tracker:

            for exp in exp_source:  
                
                trj_indx += 1 
                episode_reward = exp_source.pop_total_rewards()

                if episode_reward and tracker.reward(episode_reward[0], trj_indx):
                    break
                
                batch.append(exp)
                
                if len(batch) < GRAD_BATCH_SIZE:  #when it is filled up to batch size, then it seems the child process ends
                    continue

                data = unpack(batch, net, device=device) 
                batch.clear()
                state_tf, act_tf, exp_rew_tf = data

                net.zero_grad()
                
                logits_tf, val_s_tf = net(state_tf)
                
                loss_val = F.mse_loss(val_s_tf.squeeze(-1), exp_rew_tf)

                log_prob_tf = F.log_softmax(logits_tf, dim=1)
                
                #we use .detach() to not let PG propogate into the value approximation
                adv_tf = exp_rew_tf - val_s_tf.detach()          #Q(s,a) - v(s)
                log_prob_scale_tf = adv_tf*log_prob_tf[range(GRAD_BATCH_SIZE), act_tf]
                
                loss_policy = -log_prob_scale_tf.mean()

                prob_tf = F.softmax(logits_tf, dim=1)
                entropy_loss_tf = ENTROPY_BETA*(prob_tf*log_prob_tf).sum(dim=1).mean()

                loss_total = entropy_loss_tf + loss_val + loss_policy
                loss_total.backward()

                tb_tracker.track("advantage", adv_tf, step_indx)
                tb_tracker.track("values", val_s_tf, step_indx)
                tb_tracker.track("batch_rewards", exp_rew_tf, step_indx)
                tb_tracker.track("loss_entropy", entropy_loss_tf, step_indx)
                tb_tracker.track("loss_policy", loss_policy, step_indx)
                tb_tracker.track("loss_value", loss_val, step_indx)
                tb_tracker.track("loss_total", loss_total, step_indx)

                nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)

                grads = [
                    param.grad.data.cpu().numpy()
                    if param.grad is not None else None
                    for param in net.parameters()
                ]
                
                train_q.put(grads)
        
    train_q.put(None)

                

if __name__ == "__main__":

    mp.set_start_method('spawn')   #there are other methods, but due to PyTorch limits, this is the best option for our work
    
    #To know more about multithreading and multiproocessing: https://stackoverflow.com/questions/806499/threading-vs-parallelism-how-do-they-differ
    ##Sets the number of threads to use for parallel regions to 1
    os.environ['OMP_NUM_THREADS'] = 1  

    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("-n", '--name', required=True, help="Name of the run")
    args = parser.parse_args()

    device = "cuda" if args.cuda else "cpu"

    writer = SummaryWriter(comment="-Pong-A3C-Data_Parallel" + args.name)

    env= ptan.common.wrappers.wrap_dqn(gym.make(ENV_NAME))

    net = AtariPGN(env.observation_space.shape[0], env.action_space.n).to(device)  
    net.share_memory()   #By deafult, the weights are shared with cuda. For CPU however, we need to call this function

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)

    train_q = mp.Queue(maxsize=PROCESSES_COUNT)  #This is the Q to which each child process returns it's data to master process
    data_proc_list = []

    for proc_indx in range(PROCESSES_COUNT):
        proc_name = f"-a3c-grad_pong_{args.name}#{proc_indx}"   #here each child writes its name and records on tensorboard
        data_proc = mp.Process(target=grad_func, args=(proc_name, net, device, train_q))  #each child process
        data_proc.start()
        data_proc_list.append(data_proc)


    step_indx = 0
    grad_buffer = None


    try:
        
        while True:
            train_entry = train_q.get()
            if train_entry is None:
                break  #one of child has solved the problem

            step_indx += 1

            if grad_buffer is None:
                grad_buffer = train_entry
            else:
                for grd, new_grd in zip(grad_buffer, train_entry):
                     grd += new_grd

            if step_indx % BATCH_SIZE == 0:
                for param, grad in zip(net.parameters(), grad_buffer):
                     param.grad = torch.FloatTensor(grad).to(device)

            nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
            optimizer.step()

            grad_buffer = None

    finally:
        for p in data_proc_list:
            p.terminate()
            p.join()
