import numpy as np
from agent.trainer import Trainer
from agent.util import EpsilonExponentialDecay
from marlenv.goldmine.basic import Goldmine
from marlenv.util import GoldmineRecorder
from agent.deepq.centralized_dqn import CentralizedDQN

name = 'centralized'
exp = Experiment(name)

agent_num = 3
task_num = 4
env = Goldmine(agent_num, task_num)
observation_space = env.observation_space[0:2] + (4,)
env.seed(0)

def preprocess(obs):
    return np.concatenate([
        np.take(obs[0], [1], axis=2),  # task pos
        np.concatenate(np.take(obs, [0], axis=3), axis=2)],  # agent pos
        axis=2)

params = {
    'name'              : exp,
    'episodes'          : 40000,
    'steps'             : 200,
    'no_op_episodes'    : 100,
    'epsilon'           : EpsilonExponentialDecay(init=1.0, rate=0.9998),
    'train_every'       : 8,
    'save_model_every'  : 1000,
    'is_centralized'    : True,

    'agent_num'         : agent_num,
    'env'               : env,
    'action_space'      : env.action_space,
    'observation_space' : observation_space,
    'preprocess'        : preprocess,
    'recorder'          : GoldmineRecorder(agent_num),

    'agent':
        CentralizedDQN(
            action_space      = env.action_space,
            observation_space = observation_space,
            agent_num         = agent_num,
            memory_size       = 40000,
            batch_size        = 256,
            learning_rate     = 0.00025,
            gamma             = 0.99,
            target_update     = 200
        ),

    'hyperdash': exp
}

trainer = Trainer(**params)
trainer.train()

exp.end()
