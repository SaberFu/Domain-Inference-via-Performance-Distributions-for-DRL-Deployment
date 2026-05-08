import learning.ppo_model as ppo_model

class LCPModel(ppo_model.PPOModel):
    def __init__(self, config, env):
        super().__init__(config, env)
        return