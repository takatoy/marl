import os
import importlib
import argparse
import numpy as np
from datetime import datetime as dt

from agent.deepq.centralized_dqn import CentralizedDQN
from agent.util import Memory, EpsilonLinearDecay

from marlenv.goldmine.basic import Goldmine
from marlenv.util import GoldmineRecorder

from util import Logger

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config')
args = parser.parse_args()

# parameters
EPOCHS = 40000
STEPS = 200
NO_OP_EPOCHS = 100
MEMORY_SIZE = 40000
BATCH_SIZE = 32
LEARNING_RATE = 0.00025
GAMMA = 0.99
TARGET_UPDATE = STEPS
EPSILON = EpsilonLinearDecay(init=1.0, end=0.1, epochs=20000)
TRAIN_EVERY = 1
SAVE_MODEL_EVERY = 1000

AGENT_NUM = 3

# paths
tstr = dt.now().strftime('%Y%m%d_%H%M%S')
base_path = 'outputs/centralized_{}'.format(tstr)

if args.config and args.config != '':
    importlib.import_module(args.config)

if not os.path.exists(base_path):
    os.makedirs(base_path)

record_path = base_path + '/record'
eval_path   = base_path + '/eval'
model_path  = base_path + '/model'

reward_logger = Logger(base_path + '/rewards.log')
eval_logger   = Logger(base_path + '/eval.log')

env = Goldmine(AGENT_NUM)
agent_num = env.agent_num
action_space = env.action_space
observation_space = env.observation_space

agent = \
    CentralizedDQN(
        action_space      = action_space,
        observation_space = observation_space,
        agent_num         = agent_num,
        memory_size       = MEMORY_SIZE,
        batch_size        = BATCH_SIZE,
        learning_rate     = LEARNING_RATE,
        gamma             = GAMMA,
        target_update     = TARGET_UPDATE
    )

def preprocess(obs):
    return np.concatenate(np.array(obs), axis=0)

# Run agents with random actions to gather experience
print('Gathering random experiences...', end = '', flush=True)
for e in range(NO_OP_EPOCHS):
    obs = env.reset()
    for s in range(STEPS):
        action = np.random.choice(np.arange(action_space, dtype=np.int16), agent_num)
        nobs, reward, done, _ = env.step(action)
        agent.memory.add(preprocess(obs), action, reward, preprocess(nobs))
        obs = nobs
print(' done!', flush=True)

# Main loop
for e in range(EPOCHS):
    print('***** episode {:d} *****'.format(e))

    # Train
    print('--- train ---')
    eps = EPSILON.get(e)
    print('epsilon: {:.4f}'.format(eps))

    recorder = GoldmineRecorder(e, record_path, agent_num)

    obs = env.reset()
    recorder.record(env.render())

    total_reward = 0
    total_loss = 0.0
    train_cnt = 0
    for s in range(STEPS):
        action = agent.get_action(preprocess(obs), eps)
        nobs, reward, done, _ = env.step(np.array(action, dtype=np.int16))
        total_reward += reward.sum()

        agent.memory.add(preprocess(obs), action, reward, preprocess(nobs))
        if (s + 1) % TRAIN_EVERY == 0:
            total_loss += agent.train()
            train_cnt += 1

        obs = nobs
        recorder.record(env.render())

    recorder.close()

    ave_loss = total_loss / train_cnt
    print('total reward: {:.2f}, average loss: {:.4f}'.format(total_reward, ave_loss))
    reward_logger.log(e, total_reward, ave_loss)

    # Evaluate
    print('--- evaluate ---')

    recorder = GoldmineRecorder(e, eval_path, agent_num)
    obs = env.reset()
    recorder.record(env.render())

    total_reward = 0
    for s in range(STEPS):
        action = agent.get_action(preprocess(obs), 0.05)
        _, reward, _, _ = env.step(np.array(action, dtype=np.int16))
        total_reward += reward.sum()
        recorder.record(env.render())

    recorder.close()

    print('total reward: {:.2f}'.format(total_reward))
    eval_logger.log(e, total_reward, 0.0)

    # Save model
    if (e + 1) % SAVE_MODEL_EVERY == 0:
        agent.save(model_path, e)

    print()
