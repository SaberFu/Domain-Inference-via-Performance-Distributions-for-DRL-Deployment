import torch

import learning.lcp_model as lcp_model
import learning.ppo_agent as ppo_agent

class LCPAgent(ppo_agent.PPOAgent):
    def __init__(self, config, env, device):
        super().__init__(config, env, device)
        return

    def _load_params(self, config):
        super()._load_params(config)

        self._lcp_weight = config["lcp_weight"]
        return

    def _build_model(self, config):
        model_config = config["model"]
        self._model = lcp_model.LCPModel(model_config, self._env)
        return
    
    def _compute_actor_loss(self, batch):
        info = super()._compute_actor_loss(batch)

        norm_obs = self._obs_norm.normalize(batch["obs"])
        norm_a = self._a_norm.normalize(batch["action"])
        lcp_loss = self._compute_lcp_loss(norm_obs, norm_a)
        
        info["actor_loss"] += self._lcp_weight * lcp_loss
        info["lcp_loss"] = lcp_loss
        return info
    
    def _compute_lcp_loss(self, norm_obs, norm_a):
        norm_obs.requires_grad_(True)
        a_dist = self._model.eval_actor(norm_obs)
        a_logp = a_dist.log_prob(norm_a)

        a_logp_grad = torch.autograd.grad(a_logp, norm_obs, grad_outputs=torch.ones_like(a_logp),
                                             create_graph=True, retain_graph=True, only_inputs=True)
        a_logp_grad = a_logp_grad[0]
        a_logp_grad_norm = torch.sum(torch.square(a_logp_grad), dim=-1)

        lcp_loss = torch.mean(a_logp_grad_norm)
        return lcp_loss