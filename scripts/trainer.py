import os, sys
import shutil
import numpy as np
from datetime import datetime as dt

class Logger:
    def __init__(self, path):
        self.log_fp = open(path, 'w')
        self.log_fp.write('episode, reward, loss\n')

    def log(self, episode, reward, loss):
        self.log_fp.write('{:d},{:.6f},{:.6f}\n'.format(episode, reward, loss))

class Trainer:
    def __init__(self, env, agent, name, episodes, steps, no_op_episodes,
                 epsilon, train_every, save_model_every, agent_num, action_space,
                 recorder=None, preprocess=None, is_centralized=False):

        # Base saving path
        tstr = dt.now().strftime('%Y%m%d_%H%M%S')
        self.base_path = 'outputs/' + name + '_{}'.format(tstr)
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)

        # Copy script to record the parameters
        script_path = sys.argv[0]
        shutil.copy(script_path, self.base_path + '/script.py')

        self.record_path = self.base_path + '/record'
        if not os.path.exists(self.record_path):
            os.makedirs(self.record_path)

        self.model_path = self.base_path + '/model'
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.logger = Logger(self.base_path + '/rewards.log')

        self.env              = env
        self.agent            = agent
        self.name             = name
        self.episodes         = episodes
        self.steps            = steps
        self.no_op_episodes   = no_op_episodes
        self.epsilon          = epsilon
        self.train_every      = train_every
        self.save_model_every = save_model_every
        self.agent_num        = agent_num
        self.action_space     = action_space
        self.recorder         = recorder
        self.preprocess       = preprocess
        self.is_centralized   = is_centralized

    def train(self):
        # Run agents with random actions to gather experience
        print('Gathering random experiences...', end = '', flush=True)

        for e in range(self.no_op_episodes):
            obs = self.env.reset()
            if self.preprocess is not None:
                obs = self.preprocess(obs)

            for s in range(self.steps):
                action = np.random.choice(np.arange(self.action_space, dtype=np.int16), self.agent_num)
                nobs, reward, done, _ = self.env.step(action)

                if self.preprocess is not None:
                    nobs = self.preprocess(nobs)

                if self.is_centralized:
                    self.agent.memory.add(obs, action, reward, nobs)
                else:
                    for i, ag in enumerate(self.agent):
                        ag.memory.add(obs[i], action[i], reward[i], nobs[i])

                obs = nobs

        print(' done!', flush=True)

        # Main loop
        for e in range(self.episodes):
            print('***** episode {:d} *****'.format(e))

            ### Train ###
            print('--- train ---')
            eps = self.epsilon.get(e)
            print('epsilon: {:.4f}'.format(eps))

            self.recorder.begin_episode(self.record_path, e)

            obs = self.env.reset()
            if self.preprocess is not None:
                obs = self.preprocess(obs)

            self.recorder.record(self.env.render())

            total_reward = 0
            total_loss = 0.0
            train_cnt = 0
            for s in range(self.steps):
                if self.is_centralized:
                    action = self.agent.get_action(obs, eps)
                else:
                    action = [ag.get_action(obs[i], eps) for i, ag in enumerate(self.agent)]

                nobs, reward, done, _ = self.env.step(np.array(action, dtype=np.int16))
                total_reward += reward.sum()

                if self.preprocess is not None:
                    nobs = self.preprocess(nobs)

                if self.is_centralized:
                    self.agent.memory.add(obs, action, reward, nobs)
                    if (s + 1) % self.train_every == 0:
                        total_loss += self.agent.train()
                        train_cnt += 1
                else:
                    for i, ag in enumerate(self.agent):
                        ag.memory.add(obs[i], action[i], reward[i], nobs[i])
                        if (s + 1) % self.train_every == 0:
                            total_loss += ag.train()
                            train_cnt += 1

                obs = nobs
                self.recorder.record(self.env.render())

            self.recorder.end_episode()

            ave_loss = total_loss / train_cnt
            print('total reward: {:.2f}, average loss: {:.4f}'.format(total_reward, ave_loss))
            self.logger.log(e, total_reward, ave_loss)

            # Save model
            if (e + 1) % self.save_model_every == 0:
                if self.is_centralized:
                    self.agent.save(self.model_path, e)
                else:
                    for i, ag in enumerate(self.agent):
                        ag.save(self.model_path, e, i)

            print()

        print('Saved data at {:s}'.format(self.base_path))
