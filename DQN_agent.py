import torch.nn.functional as F
from torch.nn import init
from torch import nn
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ExponentialLR
import numpy as np
import scipy
import random, time, math
import keyboard as k
import matplotlib.pyplot as plt

from buffers import Uniform_replay_buffer


class DQN_agent:

    def __init__(self, input_space, action_space, epsilon=0.05):

        self.model = nn.Sequential(
            nn.Linear(input_space, 64),
            nn.Mish(),
            nn.Linear(64, 36),
            nn.Mish(),
            nn.Linear(36, action_space),
        )

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, a=0.0003)

        self.model.apply(init_weights)
        for param in self.model[-1].parameters():
            param.data /= 100

        self.action_space = [i for i in range(action_space)]
        self.epsilon = epsilon

    """
    inputs: state (<var> state values to be passed into conv()), valid_actions (<list> action mask of 0s and 1s)
    outputs: action (<int> index of selected action), qvals (<list> q values of all actions)
    """

    def get_action(self, state, valid_actions):
        # epsilon greedy
        qvals = self.model(self.conv(state)).detach().numpy()[0] * valid_actions
        if random.random() < self.epsilon:
            action = random.choice(self.action_space)
        else:
            action = np.argmax(qvals)
        return action, qvals

    """
    inputs: state (<var> state values to be passed into conv()), action (<int> index of selected action),
            next_state (<var> state values to be passed into conv()), next_valid_actions (<list> action mask of 0s and 1s), r (<float> reward for taking action),
            discount (<float> discount rate), scale (<float> weight of gradient)
    """

    def learn(self, state, action, next_state, next_valid_actions, r, discount):
        if next_state == -1:
            max_next_q = 0  # if end of episode next_q = 0
        else:
            # no grad to save computation
            with torch.no_grad():
                _, next_qvals = self.get_action(next_state, next_valid_actions)
                max_next_q = max(next_qvals)
        # your backward pass confused me a bit but this is a common way to do it
        # switched the order of q and next q which is the main change
        output = self.model(self.conv(state))
        td_e = 1 / 2 * (output[0, action] - r + discount * max_next_q) ** 2
        td_e.backward(retain_graph=True)  # normally you only call backward once
        # on the whole batch instead of doing this for each transition because
        # that way your cpu only has  to talk to the gpu once which speeds things
        # up imensely but since we are not doing that we need the computation graph
        # to not get deleted when we do the backward call so that it can accumulate
        # gradients from each example in the batch. Not sure if it would have been
        # overwritten here for each example but better safe than sorry
        return td_e.detach().item()

    def set_conv(self, conv):
        self.conv = conv

    def get_model(self):
        return self.model

    def get_modelparameters(self):
        return self.model.parameters()


"""
required functions in env:
    - resetEnv(), returns [state, valid_actions]
    - nextFrame(action), returns [next_state, r, valid_actions ,done]
    - convState(state), return [converted_state]

"""


class Epsilon_scheduler:

    def __init__(self, agent, min_epsilon=0.10, max_epsilon=1):
        self.agent = agent
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.step = self.default_update

    def default_update(self):
        self.agent.epsilon = max(self.agent.epsilon * 0.98, self.min_epsilon)

    def reset(self):
        self.agent.epsilon = self.max_epsilon


class DQN_trainer:

    def __init__(
        self,
        env,
        agent,
        discount=0.99,  # changed discount and batch size to common ones
        lr=0.001,
        batch_size=128,
        buffer_size=200000,
        sample_size=50000,
    ):
        self.batch_size = batch_size
        self.discount = discount
        self.agent = agent
        self.env = env
        self.agent.set_conv(self.env.convState)

        self.sample_size = sample_size
        self.replay_buffer = Uniform_replay_buffer(buffer_size, sample_size)

        self.opt = optim.RMSprop(
            self.agent.get_modelparameters(), lr=lr, weight_decay=1e-5
        )
        self.epsilon_scheduler = Epsilon_scheduler(agent)
        self.lr_scheduler = ExponentialLR(self.opt, gamma=0.98)

    def train(self, epochs, ep_len, verbose=True):
        tdes = []
        test_scores = []
        for batch in range(epochs // self.batch_size):
            print(f"{batch}: ", end="")
            self.opt.zero_grad()
            # [ state, action, max_next_q, r, next_state, next_valid_actions ]
            hist = [[], [], [], [], [], []]
            # play episode
            for ep in range(self.batch_size):
                [x.extend(y) for x, y in zip(hist, self.run_episode(ep_len))]
                if ep / self.batch_size // 0.1 > (ep - 1) / self.batch_size // 0.1:
                    print("#", end="")
            print()

            self.replay_buffer.add(hist)

            # backprop
            # indices, priorities = [], []
            tdes.append(0)
            for data in list(zip(*hist)) + self.replay_buffer.sample():
                state, action, qvals, r, next_state, next_valid_actions = data
                td_e = self.agent.learn(
                    state, action, next_state, next_valid_actions, r, self.discount
                )
                tdes[-1] += td_e
            nn.utils.clip_grad_norm_(self.agent.get_modelparameters(), 1.0)
            self.opt.step()
            self.epsilon_scheduler.step()
            # self.lr_scheduler.step()

            if verbose:
                print(tdes)
                avg_test = 0
                # print(f" test score: {self.test(ep_len,display=False,verbose=True)}")
                for _ in range(10):
                    # print(
                    #    f" test score: {self.test(ep_len,display=False)} eps {self.agent.epsilon}"
                    # )
                    avg_test += self.test(ep_len, display=False)
                print(f"avg test score: {avg_test/10}")
                test_scores.append(avg_test / 10)

        plt.plot(tdes)
        plt.title("avg td error over time")
        plt.show()
        plt.plot(test_scores)
        plt.title("Average test scores over time")
        plt.show()

    def run_episode(self, ep_len):
        # [ state, action, max_next_q, r, next_state, next_valid_actions ]
        hist = [[], [], [], [], [], []]
        state, valid_actions = self.env.resetEnv()
        done = False
        for step in range(ep_len):
            action, qvals = self.agent.get_action(state, valid_actions)
            max_q = max(qvals)
            next_state, r, next_valid_actions, done = self.env.nextFrame(action)
            [
                x.append(y)
                for x, y in zip(
                    hist, [state, action, qvals, r, next_state, next_valid_actions]
                )
            ]
            if done:
                break
            state = next_state
            valid_actions = next_valid_actions

        return hist

    def test(self, ep_len, display, verbose=False):
        done = False
        total_r = 0
        state, valid_actions = self.env.resetEnv()
        show_rest = True
        total_actions = [0 for i in range(len(self.agent.action_space))]
        for step in range(ep_len):
            action, qvals = self.agent.get_action(state, valid_actions)
            total_actions[action] += 1
            state, r, valid_actions, done = self.env.nextFrame(
                action, display=display and show_rest
            )
            if verbose and step % 20 == 0 and step <= 200:
                print(qvals, action, r, state)
            total_r += r
            if done:
                break
            if k.is_pressed("space"):
                show_rest = False
        if verbose:
            print(qvals, action, r, step)
        return total_r  # , total_actions
