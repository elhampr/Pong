import gym
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import ptan
import torch.optim as optim
import torch.nn.utils as nn_utils

import numpy as np
import argparse
import sys
import time

ENV_NAME = "PongNoFrameskip-v4"
ENV_COUNT = 32
HIDDEN_LAYER = 512
GAMMA = 0.99
LEARNING_RATE = 0.01
TRAIN_EPISODE = 4
REWARD_STEPS = 4
ENTROPY_BETA = 0.01
CLIP_GRAD = 0.1


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




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("-n", '--name', required=True, help="Name of the run")
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")
    
    writer = SummaryWriter(comment="-Pong-A2C" + args.name)

    #This enables creating of multiple parallel environments to address the correlation issue in REINFORCE/PG
    make_env= lambda: ptan.common.wrappers.wrap_dqn(gym.make(ENV_NAME))
    envs = [make_env() for _ in range(ENV_COUNT)]

    net = AtariPGN(envs[0].observation_space.shape[0], envs[0].action_space.n).to(device)
    #No need for target net
    
    #default epsilon is 1e-8 or 1e-10 to prevent zero division but in this problem default turns to be too small for convergence to occurr
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3) 
    
    #This agent uses probablity to select action based on distributions from net
    ##apply softmax to convert net output to probablity; net outputs are raw scores or logits (not probability)
    agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], device=device, apply_softmax=True) 
    
    #To know more about ExperienceSourceFirstLast: https://github.com/Shmuma/ptan/issues/17
    ##no need for replay buffer as A2C is on-policy
    ##defalut steps per episode: steps_count=2
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, gamma=GAMMA, steps_count=REWARD_STEPS) 

    batch = []

    with RewardTracker(writer, stop_reward=18) as tracker: #checks the reward with threshold
        with ptan.common.utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:

            for step_indx, exp in enumerate(exp_source):
                batch.append(exp)

                episode_reward = exp_source.pop_total_rewards()
                if episode_reward:
                    if tracker(episode_reward[0], step_indx):
                        break

                if len(batch) < TRAIN_EPISODE:
                    continue


                states, actions, exp_rew = unpack(batch, net, device=device)
                batch.clear()

                optimizer.zero_grad()
                logits_tf, val_s_tf = net(states)
                
                loss_val = F.mse_loss(val_s_tf.squeeze(-1), exp_rew)

                log_prob_tf = F.log_softmax(logits_tf, dim=1)
                #we use .detach() to not let PG propogate into the value approximation
                adv_tf = exp_rew - val_s_tf.detach()          #Q(s,a) - v(s)
                log_prob_scale_tf = adv_tf*log_prob_tf[range(TRAIN_EPISODE), actions]
                
                loss_policy = -log_prob_scale_tf.mean()

                prob_tf = F.softmax(logits_tf, dim=1)
                entropy_loss_tf = ENTROPY_BETA*(prob_tf*log_prob_tf).sum(dim=1).mean()

                # calculate policy gradients only
                loss_policy.backward(retain_graph=True)  #we retain graph since we want to recalculate loss backward later
                grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
                                        for p in net.parameters()
                                        if p.grad is not None])

                # apply entropy and value gradients
                loss_v = entropy_loss_tf + loss_val
                loss_v.backward()

                #Gradient clipping and gradient checking are two techniques that can help you avoid exploding and vanishing gradients in deep learning. 
                #Exploding gradients occur when the gradients become too large and cause numerical instability, 
                #while vanishing gradients occur when the gradients become too small and prevent learning.
                #CLIP GRAD should be choosen as a value that is large enough to allow the model to learn quickly, but small enough to prevent exploding gradients.
                nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
                optimizer.step()
                # get full loss
                loss_v += loss_policy

                
                tb_tracker.track("advantage",       adv_tf, step_indx)
                tb_tracker.track("values",          val_s_tf, step_indx)
                tb_tracker.track("batch_rewards",   exp_rew, step_indx)
                tb_tracker.track("loss_entropy",    entropy_loss_tf, step_indx)
                tb_tracker.track("loss_policy",     loss_policy, step_indx)
                tb_tracker.track("loss_value",      loss_val, step_indx)
                tb_tracker.track("loss_total",      loss_v, step_indx)
                tb_tracker.track("grad_l2",         np.sqrt(np.mean(np.square(grads))), step_indx)
                tb_tracker.track("grad_max",        np.max(np.abs(grads)), step_indx)
                tb_tracker.track("grad_var",        np.var(grads), step_indx)















