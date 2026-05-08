import numpy as np
import torch

import engines.engine as engine
import envs.smp_env as smp_env
import envs.base_env as base_env
import util.torch_util as torch_util

class TaskDodgeballEnv(smp_env.SMPEnv):
    def __init__(self, env_config, engine_config, num_envs, device, visualize, record_video=False):
        self._proj_radius = 0.1
        self._num_projectiles = int(env_config.get("num_projectiles", 1))
        self._hit_dist = float(env_config.get("hit_dist", 0.8))
        self._hit_force_threshold = float(env_config.get("hit_force_threshold", 0.1))
        self._hit_delta_v_threshold = float(env_config.get("hit_delta_v_threshold", 1.5))
        
        self._proj_dist_min = float(env_config.get("proj_dist_min", 8.0))
        self._proj_dist_max = float(env_config.get("proj_dist_max", 10.0))
        self._proj_h_min = float(env_config.get("proj_h_min", 0.5))
        self._proj_h_max = float(env_config.get("proj_h_max", 2.5))
        self._proj_h_min = max(self._proj_h_min, self._proj_radius)

        self._proj_speed_min = float(env_config.get("proj_speed_min", 12.0))
        self._proj_speed_max = float(env_config.get("proj_speed_max", 15.0))
        self._proj_trigger_time_min = float(env_config.get("proj_trigger_time_min", 1.0))
        self._proj_trigger_time_max = float(env_config.get("proj_trigger_time_max", 4.0))
        self._proj_aim_noise_scale = float(env_config.get("proj_aim_noise_scale", 0.1))
        
        assert self._num_projectiles > 0
        assert self._proj_dist_min <= self._proj_dist_max
        assert self._proj_h_min <= self._proj_h_max
        assert self._proj_speed_min <= self._proj_speed_max
        assert self._proj_trigger_time_min <= self._proj_trigger_time_max
        assert self._hit_delta_v_threshold >= 0.0

        super().__init__(env_config=env_config, engine_config=engine_config,
                         num_envs=num_envs, device=device, visualize=visualize,
                         record_video=record_video)
        return

    def _build_envs(self, config, num_envs):
        self._proj_ids = []
        super()._build_envs(config, num_envs)
        return

    def _build_env(self, env_id, config):
        super()._build_env(env_id, config)

        for proj_idx in range(self._num_projectiles):
            proj_id = self._build_projectile(env_id, proj_idx)
            if (env_id == 0):
                self._proj_ids.append(proj_id)
            else:
                assert(self._proj_ids[proj_idx] == proj_id)
        return

    def _build_projectile(self, env_id, proj_idx):
        proj_asset_file = "data/assets/objects/dodgeball.xml"
        start_pos = np.array([10.0 + proj_idx, 10.0, self._proj_radius], dtype=np.float32)
        start_rot = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        proj_id = self._engine.create_obj(env_id=env_id,
                                          obj_type=engine.ObjType.rigid,
                                          asset_file=proj_asset_file,
                                          name="projectile{:d}".format(proj_idx),
                                          start_pos=start_pos,
                                          start_rot=start_rot,
                                          color=[0.9725, 0.42, 0.1137])
        return proj_id

    def _build_sim_tensors(self, config):
        super()._build_sim_tensors(config)

        num_envs = self.get_num_envs()
        self._proj_trigger_times = torch.zeros([num_envs, self._get_num_projectiles()],
                                               device=self._device, dtype=torch.float)
        self._prev_proj_vel = torch.zeros([num_envs, self._get_num_projectiles(), 3],
                                          device=self._device, dtype=torch.float)
        
        proj_target_body = config["proj_target_body"]
        self._proj_target_body_id = int(self._build_body_ids_tensor([proj_target_body])[0].item())
        return

    def _get_num_projectiles(self):
        return len(self._proj_ids)

    def _pre_physics_step(self, actions):
        super()._pre_physics_step(actions)
        self._record_proj_vel()
        return

    def _record_proj_vel(self):
        _, proj_vel = self._get_proj_states()
        self._prev_proj_vel[:] = proj_vel
        return

    def _sample_proj_trigger_dt(self, n, num_projectiles):
        rand_dt = torch.rand([n, num_projectiles], device=self._device, dtype=torch.float)
        rand_dt = (self._proj_trigger_time_max - self._proj_trigger_time_min) * rand_dt + self._proj_trigger_time_min
        return rand_dt

    def _build_inactive_proj_pos(self, n, proj_idx):
        pos = torch.zeros([n, 3], device=self._device, dtype=torch.float)
        pos[:, 0] = 10.0 + proj_idx
        pos[:, 1] = 10.0
        pos[:, 2] = self._proj_radius
        return pos

    def _get_proj_states(self):
        proj_pos = []
        proj_vel = []
        for proj_id in self._proj_ids:
            proj_pos.append(self._engine.get_root_pos(proj_id))
            proj_vel.append(self._engine.get_root_vel(proj_id))

        proj_pos = torch.stack(proj_pos, dim=1)
        proj_vel = torch.stack(proj_vel, dim=1)
        return proj_pos, proj_vel

    def _get_proj_contact_force(self):
        proj_contact_force = []
        for proj_id in self._proj_ids:
            contact_force = self._engine.get_contact_forces(proj_id)
            contact_force_xy = torch.linalg.norm(contact_force[..., 0:2], dim=-1)
            contact_force_xy = torch.amax(contact_force_xy, dim=-1)
            proj_contact_force.append(contact_force_xy)

        proj_contact_force = torch.stack(proj_contact_force, dim=1)
        return proj_contact_force

    def _update_task(self):
        trigger_mask = self._time_buf.unsqueeze(-1) >= self._proj_trigger_times
        trigger_ids = trigger_mask.nonzero(as_tuple=False)

        if (trigger_ids.shape[0] > 0):
            self._launch_projectiles(env_ids=trigger_ids[:, 0], proj_ids=trigger_ids[:, 1])
        return

    def _update_misc(self):
        super()._update_misc()
        self._update_task()
        return

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)

        if (len(env_ids) > 0):
            self._reset_projectiles(env_ids)
        return

    def _reset_projectiles(self, env_ids):
        n = env_ids.shape[0]
        reset_rot = torch.zeros([n, 4], device=self._device, dtype=torch.float)
        reset_rot[:, 3] = 1.0
        reset_vel = torch.zeros([n, 3], device=self._device, dtype=torch.float)

        for proj_idx, proj_id in enumerate(self._proj_ids):
            reset_pos = self._build_inactive_proj_pos(n, proj_idx)
            self._engine.set_root_pos(env_ids, proj_id, reset_pos)
            self._engine.set_root_rot(env_ids, proj_id, reset_rot)
            self._engine.set_root_vel(env_ids, proj_id, reset_vel)
            self._engine.set_root_ang_vel(env_ids, proj_id, reset_vel)
            self._prev_proj_vel[env_ids, proj_idx] = reset_vel

        rand_dt = self._sample_proj_trigger_dt(n, self._get_num_projectiles())
        self._proj_trigger_times[env_ids] = self._time_buf[env_ids].unsqueeze(-1) + rand_dt
        return

    def _compute_ballistic_launch_state(self, target_pos, target_vel, launch_speed):
        n = target_pos.shape[0]
        gravity = self._engine.get_gravity()
        rand_h = (self._proj_h_max - self._proj_h_min) * torch.rand(n, device=self._device) + self._proj_h_min
        rand_dist = (self._proj_dist_max - self._proj_dist_min) * torch.rand(n, device=self._device) + self._proj_dist_min
        travel_time = rand_dist / launch_speed

        dh = target_pos[:, 2] - rand_h
        vel_z = dh / travel_time - 0.5 * gravity[2] * travel_time

        target_pred_pos = target_pos + target_vel * travel_time.unsqueeze(-1)
        
        rand_theta = 2 * np.pi * torch.rand(n, device=self._device)
        proj_pos = torch.zeros([n, 3], device=self._device, dtype=torch.float)
        proj_pos[:, 0] = target_pred_pos[:, 0] + rand_dist * torch.cos(rand_theta)
        proj_pos[:, 1] = target_pred_pos[:, 1] + rand_dist * torch.sin(rand_theta)
        proj_pos[:, 2] = rand_h

        proj_delta = target_pred_pos - proj_pos
        proj_delta[:, 2] = 0
        proj_dir = torch.nn.functional.normalize(proj_delta, dim=-1)
        proj_vel = launch_speed.unsqueeze(-1) * proj_dir
        proj_vel[:, 2] = vel_z

        return proj_pos, proj_vel

    def _launch_projectiles(self, env_ids, proj_ids):
        n = env_ids.shape[0]
        if (n > 0):
            char_id = self._get_char_id()
            char_body_pos = self._engine.get_body_pos(char_id)[env_ids]
            char_body_vel = self._engine.get_body_vel(char_id)[env_ids]

            target_pos = char_body_pos[:, self._proj_target_body_id, :].clone()
            target_pos += self._proj_aim_noise_scale * torch.randn_like(target_pos)
            target_vel = char_body_vel[:, self._proj_target_body_id, :].clone()

            launch_speed = (self._proj_speed_max - self._proj_speed_min) * torch.rand([n], device=self._device) + self._proj_speed_min
            launch_pos, launch_vel = self._compute_ballistic_launch_state(target_pos=target_pos,
                                                                        target_vel=target_vel,
                                                                        launch_speed=launch_speed)

            launch_rot = torch.zeros([n, 4], device=self._device, dtype=torch.float)
            launch_rot[:, 3] = 1.0
            launch_ang_vel = torch.zeros([n, 3], device=self._device, dtype=torch.float)

            for local_proj_id, proj_obj_id in enumerate(self._proj_ids):
                curr_mask = proj_ids == local_proj_id

                if (curr_mask.any().item()):
                    curr_env_ids = env_ids[curr_mask]
                    self._engine.set_root_pos(curr_env_ids, proj_obj_id, launch_pos[curr_mask])
                    self._engine.set_root_rot(curr_env_ids, proj_obj_id, launch_rot[curr_mask])
                    self._engine.set_root_vel(curr_env_ids, proj_obj_id, launch_vel[curr_mask])
                    self._engine.set_root_ang_vel(curr_env_ids, proj_obj_id, launch_ang_vel[curr_mask])
                    self._prev_proj_vel[curr_env_ids, local_proj_id] = launch_vel[curr_mask]

            rand_dt = self._sample_proj_trigger_dt(n, 1).squeeze(-1)
            self._proj_trigger_times[env_ids, proj_ids] = self._time_buf[env_ids] + rand_dt
        return

    def _compute_obs(self, env_ids=None):
        obs = super()._compute_obs(env_ids)

        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)
        root_rot = self._engine.get_root_rot(char_id)
        proj_pos, proj_vel = self._get_proj_states()

        if (env_ids is not None):
            root_pos = root_pos[env_ids]
            root_rot = root_rot[env_ids]
            proj_pos = proj_pos[env_ids]
            proj_vel = proj_vel[env_ids]

        task_obs = compute_dodgeball_observations(root_pos=root_pos,
                                              root_rot=root_rot,
                                              proj_pos=proj_pos,
                                              proj_vel=proj_vel)
        obs = torch.cat([obs, task_obs], dim=-1)
        return obs

    def _update_reward(self):
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)
        root_vel = self._engine.get_root_vel(char_id)
        proj_pos, _ = self._get_proj_states()

        self._reward_buf[:] = compute_dodge_reward(root_pos=root_pos, root_vel=root_vel, proj_pos=proj_pos)
        return

    def _update_done(self):
        super()._update_done()
        
        if (self._enable_early_termination):
            char_id = self._get_char_id()
            body_pos = self._engine.get_body_pos(char_id)
            proj_pos, proj_vel = self._get_proj_states()
            proj_contact_force = self._get_proj_contact_force()
            gravity = self._engine.get_gravity()

            hit_fail = compute_dodgeball_fail_flags(body_pos=body_pos,
                                                    proj_pos=proj_pos,
                                                    proj_vel=proj_vel,
                                                    prev_proj_vel=self._prev_proj_vel,
                                                    proj_contact_force=proj_contact_force,
                                                    hit_dist=self._hit_dist,
                                                    hit_force_threshold=self._hit_force_threshold,
                                                    hit_delta_v_threshold=self._hit_delta_v_threshold,
                                                    gravity=gravity[2],
                                                    dt=self._engine.get_timestep())
            self._done_buf[hit_fail] = base_env.DoneFlags.FAIL.value
        return


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_dodgeball_observations(root_pos, root_rot, proj_pos, proj_vel):
    # type: (Tensor, Tensor, Tensor, Tensor) -> Tensor
    heading_inv_rot = torch_util.calc_heading_quat_inv(root_rot)
    heading_inv_rot = heading_inv_rot.unsqueeze(-2).repeat((1, proj_pos.shape[1], 1))

    proj_relative_pos = proj_pos - root_pos.unsqueeze(-2)
    local_proj_pos = torch_util.quat_rotate(heading_inv_rot, proj_relative_pos)
    local_proj_vel = torch_util.quat_rotate(heading_inv_rot, proj_vel)
    
    proj_obs = torch.cat([local_proj_pos, local_proj_vel], dim=-1)
    proj_obs_flat = proj_obs.reshape(proj_obs.shape[0], -1)

    obs = proj_obs_flat
    return obs

@torch.jit.script
def compute_dodgeball_fail_flags(body_pos, proj_pos, proj_vel, prev_proj_vel, proj_contact_force,
                                 hit_dist, hit_force_threshold,
                                 hit_delta_v_threshold, gravity, dt):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, float, float, float, float) -> Tensor

    proj_body_delta = proj_pos.unsqueeze(-2) - body_pos.unsqueeze(-3)
    proj_around_body = torch.linalg.norm(proj_body_delta, dim=-1) < hit_dist
    proj_around_body = torch.any(proj_around_body, dim=-1)

    expected_proj_vel = prev_proj_vel.clone()
    expected_proj_vel[..., 2] += gravity * dt
    proj_delta_v = torch.linalg.norm(proj_vel - expected_proj_vel, dim=-1)

    hit_detected = proj_contact_force > hit_force_threshold
    hit_detected = torch.logical_and(hit_detected, proj_around_body)

    delta_v_hit = proj_delta_v > hit_delta_v_threshold
    delta_v_hit = torch.logical_and(delta_v_hit, proj_around_body)
    hit_detected = torch.logical_or(hit_detected, delta_v_hit)
    hit_fail = torch.any(hit_detected, dim=-1)

    return hit_fail

@torch.jit.script
def compute_dodge_reward(root_pos, root_vel, proj_pos):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    pos_w = 0.9
    vel_w = 0.1

    pos_scale = 0.3
    vel_scale = 1.0

    pos_diff = proj_pos - root_pos.unsqueeze(-2)
    pos_err = torch.min(torch.linalg.norm(pos_diff, dim=-1), dim=-1).values
    pos_reward = 1.0 - torch.exp(-pos_scale * pos_err)

    vel_err = torch.sum(torch.square(root_vel[..., :2]), dim=-1)
    vel_reward = torch.exp(-vel_scale * vel_err)

    reward = pos_w * pos_reward + vel_w * vel_reward

    return reward