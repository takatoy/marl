class Env:
    # Set these in ALL subclasses
    agent_num = 0            # Number of agents
    action_space = 0         # Number of action types
    observation_space = None # Shape of observation for SINGLE agent

    def step(self, actions):
        """
        Run one step in the environment.

        Args:
            actions (object): Shape of this must be (agent_num,) and
                every single value must be within the range of [0, action_space - 1]

        Returns:
            observation (object): Shape of this will be (agent_num,) + observation_space
            reward (float)
            done (boolean)
            info (dict)
        """
        raise NotImplementedError

    def reset(self):
        """
        Reset the environment.

        Returns: observation(object): Shape of this will be (agent_num,) + observation_space
        """
        raise NotImplementedError

    def render(self):
        """
        Render the environment.
        """
        raise NotImplementedError

    def close(self):
        """
        Run for cleaning up environment.
        """
        pass

    def seed(self, seed=None):
        """
        Set random seed.
        """
        pass
