import os
import numpy as np
from datetime import datetime as dt
from marlenv.util import GoldmineRecorder

class Logger:
    def __init__(self, path):
        self.log_fp = open(path, 'w')

    def log(self, epoch, reward, loss):
        self.log_fp.write('epoch {:d}: reward {:.6f}, loss {:.6f}\n'.format(epoch, reward, loss))

class Trainer:
    def __init__(self, env, agent, name, episodes, steps, no_op_episodes,
                 epsilon, train_every, save_model_every, agent_num, action_space,
                 preprocess=None, is_centralized=False):

        # Prepare saving paths
        tstr = dt.now().strftime('%Y%m%d_%H%M%S')
        base_path = 'outputs/' + name + '_{}'.format(tstr)
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        self.record_path = base_path + '/record'
        if not os.path.exists(self.record_path):
            os.makedirs(self.record_path)
        self.eval_path = base_path + '/eval'
        if not os.path.exists(self.eval_path):
            os.makedirs(self.eval_path)
        self.model_path = base_path + '/model'
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.logger = Logger(base_path + '/rewards.log')

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

            recorder = GoldmineRecorder(e, self.record_path, self.agent_num)

            obs = self.env.reset()
            if self.preprocess is not None:
                obs = self.preprocess(obs)

            recorder.record(self.env.render())

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
                recorder.record(self.env.render())

            recorder.close()

            ave_loss = total_loss / train_cnt
            print('total reward: {:.2f}, average loss: {:.4f}'.format(total_reward, ave_loss))
            self.logger.log(e, total_reward, ave_loss)

            # Save model
            if (e + 1) % self.save_model_every == 0:
                if self.is_centralized:
                    self.agent.save(model_path, e)
                else:
                    for i, ag in enumerate(self.agent):
                        ag.save(model_path + '_{}'.format(i), e)

            print()
