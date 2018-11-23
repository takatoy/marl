from agent.trainer import Trainer
from agent.util import EpsilonLinearDecay
from marlenv.goldmine.relative import GoldmineRV
from marlenv.util import GoldmineRecorder
from agent.deepq.light_dqn import LightDQN

agent_num = 6
task_num = 4
view_range = 3
env = GoldmineRV(agent_num, task_num, view_range)

params = {
    'name': 'rv_light',
    'episodes': 40000,
    'steps': 200,
    'no_op_episodes': 100,
    'epsilon': EpsilonLinearDecay(init=1.0, end=0.05, episodes=5000),
    'train_every': 8,
    'save_model_every': 1000,
    'is_centralized': False,

    'agent_num': agent_num,
    'env': env,
    'action_space': env.action_space,
    'observation_space': env.observation_space,
    'preprocess': None,
    'recorder': GoldmineRecorder(agent_num),

    'agent': [
        LightDQN(
            action_space=env.action_space,
            observation_space=env.observation_space,
            memory_size=40000,
            batch_size=256,
            learning_rate=0.00025,
            gamma=0.99,
            target_update=200,
            use_dueling=False
        ) for _ in range(agent_num)
    ]
}

trainer = Trainer(**params)
trainer.train()
