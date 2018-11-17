from agent.trainer import Trainer
from agent.util import EpsilonExponentialDecay
from marlenv.goldmine.memorize import GoldmineMV
from marlenv.util import GoldmineRecorder
from agent.deepq.miyashita_dqn import MiyashitaDQN

agent_num = 6
task_num = 4
view_range = 3
mem_period = 10
env = GoldmineMV(agent_num, task_num, view_range, mem_period)

params = {
    'name'              : 'mv_miyashita_env',
    'episodes'          : 40000,
    'steps'             : 200,
    'no_op_episodes'    : 100,
    'epsilon'           : EpsilonExponentialDecay(init=1.0, rate=0.9999),
    'train_every'       : 1,
    'save_model_every'  : 1000,
    'is_centralized'    : False,

    'agent_num'         : agent_num,
    'env'               : env,
    'action_space'      : env.action_space,
    'preprocess'        : None,
    'recorder'          : GoldmineRecorder(agent_num),

    'agent': [
        MiyashitaDQN(
            action_space      = env.action_space,
            observation_space = env.observation_space,
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
