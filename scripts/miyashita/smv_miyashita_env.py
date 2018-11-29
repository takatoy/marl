import numpy as np
from agent.trainer import Trainer
from agent.util import EpsilonExponentialDecay
from marlenv.goldmine.relative import GoldmineRV
from marlenv.util import GoldmineRecorder
from agent.deepq.miyashita_dqn import MiyashitaDQN

agent_num = 6
task_num = 4
view_range = 3
env = GoldmineRV(agent_num, task_num, view_range)
env.seed(0)
obs_num = 3
observation_space = env.observation_space[0:2] + (env.observation_space[2] * obs_num,)

def preprocess(obs):
    n = len(obs)
    pr_obs = np.empty((n,) + observation_space)
    for i, o in enumerate(obs):
        pr_obs[i] = np.dstack(o)
    return pr_obs

params = {
    'name'              : 'smv_miyashita_env',
    'episodes'          : 40000,
    'steps'             : 200,
    'no_op_episodes'    : 100,
    'epsilon'           : EpsilonExponentialDecay(init=1.0, rate=0.9999),
    'train_every'       : 1,
    'save_model_every'  : 1000,
    'is_centralized'    : False,
    'obs_num'           : obs_num,

    'agent_num'         : agent_num,
    'env'               : env,
    'action_space'      : env.action_space,
    'observation_space' : env.observation_space,
    'preprocess'        : preprocess,
    'recorder'          : GoldmineRecorder(agent_num),

    'agent': [
        MiyashitaDQN(
            action_space      = env.action_space,
            observation_space = observation_space,
            memory_size       = 2000,
            batch_size        = 32,
            learning_rate     = 0.0001,
            gamma             = 0.99,
            target_update     = 200
        ) for _ in range(agent_num)
    ]
}

trainer = Trainer(**params)
trainer.train()
