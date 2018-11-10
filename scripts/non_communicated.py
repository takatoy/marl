from trainer import Trainer
from agent.util import EpsilonLinearDecay
from marlenv.goldmine.basic import Goldmine
from agent.deepq.simple_dqn import SimpleDQN

agent_num = 6
env = Goldmine(agent_num)

params = {
    'name'              : 'non_communicated',
    'episodes'          : 40000,
    'steps'             : 200,
    'no_op_episodes'    : 100,
    'epsilon'           : EpsilonLinearDecay(init=1.0, end=0.05, epochs=5000),
    'train_every'       : 8,
    'save_model_every'  : 1000,
    'is_centralized'    : False,

    'agent_num'         : agent_num,
    'env'               : env,
    'action_space'      : env.action_space,
    'preprocess'        : None,

    'agent': [
        SimpleDQN(
            action_space      = env.action_space,
            observation_space = env.observation_space,
            memory_size       = 40000,
            batch_size        = 256,
            learning_rate     = 0.00025,
            gamma             = 0.99,
            target_update     = 200,
            use_dueling       = False
        ) for _ in range(agent_num)
    ]
}

trainer = Trainer(**params)
trainer.train()
