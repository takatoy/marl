import os
import numpy as np
from datetime import datetime as dt

from agent.deepq.non_communicated_dqn import NonCommunicatedDQN
from agent.util import Memory, EpsilonLinearDecay

from marlenv.goldmine.basic import Goldmine
from marlenv.util import GoldmineRecorder

from util import Logger

# parameters
EPOCHS = 40000
STEPS = 200
NO_OP_EPOCHS = 100
MEMORY_SIZE = 40000
BATCH_SIZE = 32
LEARNING_RATE = 0.00025
GAMMA = 0.99
TARGET_UPDATE = STEPS
USE_DUAL = False
EPSILON = EpsilonLinearDecay(init=1.0, end=0.1, epochs=20000)
TRAIN_EVERY = 1
SAVE_MODEL_EVERY = 1000

AGENT_NUM = 3

# paths
tstr = dt.now().strftime('%Y%m%d_%H%M%S')
base_path = 'outputs/ncdqn_{}'.format(tstr)

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

agents = [
    NonCommunicatedDQN(
        action_space      = action_space,
        observation_space = observation_space,
        memory_size       = MEMORY_SIZE,
        batch_size        = BATCH_SIZE,
        learning_rate     = LEARNING_RATE,
        gamma             = GAMMA,
        target_update     = TARGET_UPDATE,
        use_dual          = USE_DUAL
    ) for _ in range(agent_num)
]

# Run agents with random actions to gather experience
print('Gathering random experiences...', end = '', flush=True)
for e in range(NO_OP_EPOCHS):
    obs = env.reset()
    for s in range(STEPS):
        action = np.random.choice(np.arange(action_space, dtype=np.int16), agent_num)
        nobs, reward, done, _ = env.step(action)
        for i, agent in enumerate(agents):
            agent.memory.add(obs[i], action[i], reward[i], nobs[i])
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
        action = [agent.get_action(obs[i], eps) for i, agent in enumerate(agents)]
        nobs, reward, done, _ = env.step(np.array(action, dtype=np.int16))
        total_reward += reward.sum()

        for i, agent in enumerate(agents):
            agent.memory.add(obs[i], action[i], reward[i], nobs[i])
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
        action = [agent.get_action(obs[i], 0.05) for i, agent in enumerate(agents)]  # Greedy
        _, reward, _, _ = env.step(np.array(action, dtype=np.int16))
        total_reward += reward.sum()
        recorder.record(env.render())

    recorder.close()

    print('total reward: {:.2f}'.format(total_reward))
    eval_logger.log(e, total_reward, 0.0)

    # Save model
    if (e + 1) % SAVE_MODEL_EVERY == 0:
        for agent in agents:
            agent.save(model_path, e)

    print()
