import os
import numpy as np
import torch
import yaml

import learning.smp_model as smp_model
import learning.diff_normalizer as diff_normalizer
import learning.ppo_agent as ppo_agent

from learning.tinymdm.tinymdm_model import TinyMDMModel

class SMPAgent(ppo_agent.PPOAgent):
    def __init__(self, config, env, device):
        super().__init__(config, env, device)

        if self._enable_gsi:
            self._check_gsi_env_compatibility()
            self._init_gsi_buffer()
        return
    
    def _load_params(self, config):
        super()._load_params(config)

        self._sds_loss_scale = config["sds_loss_scale"]
        self._smp_reward_scale = config.get("smp_reward_scale", 1.0)
        self._smp_eval_batch_size = config["smp_eval_batch_size"]
        self._diffusion_steps = config.get("diffusion_steps", None)
        self._sds_normalizer_samples = config.get("sds_normalizer_samples", np.inf)

        self._enable_gsi = config.get("enable_gsi", False)
        self._gsi_iters = config.get("gsi_iters", 50)
        self._gsi_sampler = config.get("gsi_sampler", "ddpm")
        self._gsi_inference_steps = config.get("gsi_inference_steps", 10)
        self._gsi_buffer_size = config.get("gsi_buffer_size", 4096)
        self._gsi_regen_num_motions = config.get("gsi_regen_num_motions", 1024)
        self._gsi_batch_size = config.get("gsi_batch_size", 256)
        assert(self._gsi_buffer_size >= self._gsi_regen_num_motions)
        assert(self._gsi_regen_num_motions >= self._gsi_batch_size)
        
        self._task_reward_weight = config["task_reward_weight"]
        self._smp_reward_weight = config["smp_reward_weight"]
        return

    def _build_model(self, config):
        model_config = config["model"]
        self._model = smp_model.SMPModel(model_config, self._env)
        self._build_prior_model(config)
        return
    
    def _build_prior_model(self, agent_config):
        smp_cfg_path = agent_config["smp_prior_cfg"]
        model_path = agent_config["smp_prior_model"]
        assert os.path.isfile(smp_cfg_path), "Missing SMP prior config: {}".format(smp_cfg_path)
        assert os.path.isfile(model_path), "Missing SMP prior model: {}".format(model_path)

        with open(smp_cfg_path, "r") as stream:
            config = yaml.safe_load(stream)
        
        self._check_prior_env_config(config)
        
        disc_obs_space = self._env.get_disc_obs_space()

        config["input_dim"] = disc_obs_space.shape[-1]
        self._prior_model = TinyMDMModel(config)

        prior_state_dict = torch.load(model_path, map_location=self._device)
        incompatible_keys = self._prior_model.load_state_dict(prior_state_dict)
        if (len(incompatible_keys[0]) > 0 or len(incompatible_keys[1]) > 0):
            print("loading prior model with incompatible_keys:", incompatible_keys)

        print(
            "Loaded SMP prior model:",
            f"cfg={smp_cfg_path},",
            f"model={model_path}"
        )

        self._prior_model.eval()
        self._prior_model.to(self._device)

        for p in self._prior_model.parameters():
            p.requires_grad = False
        
        return

    def _check_prior_env_config(self, prior_config):
        with open(prior_config["env_config"], "r") as stream:
            prior_env_config = yaml.safe_load(stream)
       
        env_checks = [
            ("global_obs", self._env._global_obs, False),
            ("root_height_obs", self._env._root_height_obs, False),
            ("enable_tar_obs", self._env._enable_tar_obs, False),
            ("num_disc_obs_steps", self._env._num_disc_obs_steps, None),
            ("disc_dof_vel_obs", self._env._disc_dof_vel_obs, False),
        ]

        for key, env_val, default_val in env_checks:
            prior_val = prior_env_config.get(key, default_val)
            assert prior_val == env_val, \
                "SMP prior env mismatch for {}: prior={}, env={}".format(key, prior_val, env_val)

        prior_key_bodies = prior_env_config.get("key_bodies", [])
        env_num_key_bodies = len(self._env._key_body_ids)
        assert len(prior_key_bodies) == env_num_key_bodies, \
            "SMP prior env mismatch for key_bodies: prior={}, env={}".format(len(prior_key_bodies), env_num_key_bodies)

        prior_control_freq = prior_config.get("control_freq", None)
        env_control_freq = int(round(1.0 / self._env._engine.get_timestep()))
        assert prior_control_freq == env_control_freq, \
            "SMP prior config mismatch for control_freq: prior={}, env={}".format(prior_control_freq, env_control_freq)
        return

    def _build_normalizers(self):
        super()._build_normalizers()
        self._sds_normalizer = diff_normalizer.DiffNormalizer([len(self._diffusion_steps)], device=self._device, dtype=torch.float32)
        return

    def _train_iter(self):
        info = super()._train_iter()

        if (self._need_sds_normalizer_update()):
            self._update_sds_normalizers()

        if self._need_gsi_update():
            self._update_gsi_buffer()
        
        return info
    
    def _record_data_post_step(self, next_obs, r, done, next_info):
        super()._record_data_post_step(next_obs, r, done, next_info)

        disc_obs = next_info["disc_obs"]
        self._exp_buffer.record("disc_obs", disc_obs)
        return

    def _need_sds_normalizer_update(self):
        return self._sample_count < self._sds_normalizer_samples

    def _check_gsi_env_compatibility(self):
        assert hasattr(self._env, "init_gsi_buffer"), \
            "GSI init-state buffer only supports SMP envs with init_gsi_buffer()."
        assert not getattr(self._env, "_enable_tar_obs", False), \
            "SMP GSI init-state buffer requires enable_tar_obs=False."
        assert not getattr(self._env, "_pose_termination", False), \
            "SMP GSI init-state buffer requires pose_termination=False."
        return
    
    def _need_gsi_update(self):
        return self._enable_gsi and (self._iter % self._gsi_iters == 0)

    def _update_sds_normalizers(self):
        self._sds_normalizer.update()
        return

    def _build_train_data(self):
        reward_info = self._compute_rewards()
        
        info = super()._build_train_data()
        info = {**info, **reward_info}
        return info

    def _compute_rewards(self):
        task_r = self._exp_buffer.get_data_flat("reward")
        disc_obs = self._exp_buffer.get_data_flat("disc_obs")

        batch_size = disc_obs.shape[0]
        disc_obs_reshaped = disc_obs.reshape(batch_size, self._env._num_disc_obs_steps, -1)
        norm_disc_obs = self._prior_model.normalize(disc_obs_reshaped)
        norm_disc_obs_flat = norm_disc_obs.reshape(batch_size, -1)
        smp_r, sds_info = self._calc_smp_rewards(norm_disc_obs_flat)
        
        smp_reward_std, smp_reward_mean = torch.std_mean(smp_r)

        r = self._task_reward_weight * task_r + self._smp_reward_weight * smp_r
        self._exp_buffer.set_data_flat("reward", r)

        info = {
            "smp_reward_mean": smp_reward_mean,
            "smp_reward_std": smp_reward_std
        }
        info.update(sds_info)

        return info

    def _calc_smp_rewards(self, norm_disc_obs):
        n = norm_disc_obs.shape[0]

        smp_r_list = []
        mean_sds_loss_list = []

        with torch.no_grad():
            for i in range(0, n, self._smp_eval_batch_size):
                sds_losses = self._prior_model.ESM_SDS_loss(
                    norm_x_obs=norm_disc_obs[i:i+self._smp_eval_batch_size],
                    t_lst=self._diffusion_steps,
                )

                if (self._need_sds_normalizer_update()):
                    self._sds_normalizer.record(sds_losses) # shape: (#timesteps)

                sds_losses_norm = self._sds_normalizer.normalize(sds_losses) # (mini_bsz, #timesteps)
                mean_sds_loss_norm = torch.mean(sds_losses_norm, dim=-1)  # (mini_bsz)

                smp_r = torch.exp(-mean_sds_loss_norm * self._sds_loss_scale)
                smp_r = smp_r * self._smp_reward_scale

                mean_sds_loss = torch.mean(sds_losses, dim=-1)
                mean_sds_loss_list.append(mean_sds_loss)
                smp_r_list.append(smp_r)

            smp_r = torch.cat(smp_r_list, dim=0)
            mean_sds_loss = torch.cat(mean_sds_loss_list, dim=0)

            sds_info = dict()
            sds_info["sds_loss_mean"] = torch.mean(mean_sds_loss)
            sds_info["sds_loss_std"] = torch.std(mean_sds_loss)
        
        return smp_r, sds_info

    def _init_gsi_buffer(self):
        gsi_samples = self._generate_init_states(self._gsi_buffer_size)
        self._env.init_gsi_buffer(gsi_samples)
        return
    
    def _update_gsi_buffer(self):
        gsi_samples = self._generate_init_states(self._gsi_regen_num_motions)
        self._env.add_gsi_samples(gsi_samples)
        return

    @torch.no_grad()
    def _generate_init_states(self, num_motions):
        disc_obs_space = self._env.get_disc_obs_space()
        num_steps = self._env._num_disc_obs_steps
        batch_size = self._gsi_batch_size

        if self._prior_model.model_ema:
            gen_func = self._prior_model.sample_ema
        else:
            gen_func = self._prior_model.sample

        gsi_samples = []

        for j in range(0, num_motions, batch_size):
            curr_batch_size = min(batch_size, num_motions - j)
            norm_samples = gen_func(
                shape=disc_obs_space.shape,
                batch_size=curr_batch_size,
                device=self._device,
                sampler=self._gsi_sampler,
                num_inference_steps=self._gsi_inference_steps,
            )
            norm_samples = norm_samples.reshape(curr_batch_size, num_steps, -1)
            curr_samples = self._prior_model.unnormalize(norm_samples)
            gsi_samples.append(curr_samples)

        gsi_samples = torch.cat(gsi_samples, dim=0)
        return gsi_samples