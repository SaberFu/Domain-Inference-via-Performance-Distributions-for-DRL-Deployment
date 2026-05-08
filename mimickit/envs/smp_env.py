import torch

import envs.amp_env as amp_env
import learning.experience_buffer as experience_buffer
import util.torch_util as torch_util

class SMPEnv(amp_env.AMPEnv):
    def __init__(self, env_config, engine_config, num_envs, device, visualize, record_video=False):
        super().__init__(env_config=env_config, 
                         engine_config=engine_config, 
                         num_envs=num_envs, 
                         device=device,
                         visualize=visualize, 
                         record_video=record_video)

        self._gsi_buffer = None
        return

    def init_gsi_buffer(self, gsi_samples):
        gsi_buffer_size = gsi_samples.shape[0]
        assert gsi_buffer_size > 0

        self._gsi_buffer = experience_buffer.ExperienceBuffer(buffer_length=gsi_buffer_size, batch_size=1,
                                                              device=self._device)
        self.add_gsi_samples(gsi_samples)
        return

    def add_gsi_samples(self, gsi_samples):
        gsi_frames = self._disc_obs_to_motion_frames(gsi_samples)
        gsi_states = self._motion_frames_to_init_states(gsi_frames)
        self._gsi_buffer.push(gsi_states)
        return
    
    def _reset_ref_motion(self, env_ids):
        if (self._has_gsi_buffer()):
            self._reset_ref_motion_gsi(env_ids)
        else:
            super()._reset_ref_motion(env_ids)
        return

    def _reset_disc_hist(self, env_ids):
        if (not self._has_gsi_buffer()):
            super()._reset_disc_hist(env_ids)
        return

    def _has_gsi_buffer(self):
        return self._gsi_buffer is not None
    
    def _reset_ref_motion_gsi(self, env_ids):
        n = len(env_ids)
        motion_ids = self._motion_lib.sample_motions(n)
        self._motion_ids[env_ids] = motion_ids
        self._motion_time_offsets[env_ids] = 0.0

        gsi_states = self._gsi_buffer.sample(n)
        
        self._ref_root_pos[env_ids] = gsi_states["root_pos"][:, -1]
        self._ref_root_rot[env_ids] = gsi_states["root_rot"][:, -1]
        self._ref_root_vel[env_ids] = gsi_states["root_vel"][:, -1]
        self._ref_root_ang_vel[env_ids] = gsi_states["root_ang_vel"][:, -1]
        self._ref_joint_rot[env_ids] = gsi_states["joint_rot"][:, -1]
        self._ref_dof_pos[env_ids] = gsi_states["dof_pos"][:, -1]
        self._ref_dof_vel[env_ids] = gsi_states["dof_vel"][:, -1]
        self._ref_body_pos[env_ids] = gsi_states["body_pos"][:, -1]
        self._ref_body_rot[env_ids] = gsi_states["body_rot"][:, -1]

        self._disc_hist_root_pos.fill(env_ids, gsi_states["root_pos"])
        self._disc_hist_root_rot.fill(env_ids, gsi_states["root_rot"])
        self._disc_hist_root_vel.fill(env_ids, gsi_states["root_vel"])
        self._disc_hist_root_ang_vel.fill(env_ids, gsi_states["root_ang_vel"])
        self._disc_hist_joint_rot.fill(env_ids, gsi_states["joint_rot"])
        self._disc_hist_dof_vel.fill(env_ids, gsi_states["dof_vel"])
        self._disc_hist_body_pos.fill(env_ids, gsi_states["body_pos"])
        return

    def _disc_obs_to_motion_frames(self, sample):
        kin_char_model = self._kin_char_model
        num_rots = kin_char_model.get_num_joints() - 1
        frame_size = self._motion_lib.get_motion_frame_size()
        
        data_frames = torch.zeros(list(sample.shape[:-1]) + [frame_size], device=sample.device, dtype=sample.dtype)

        root_pos_obs = sample[..., 0:3]
        root_rot_obs = sample[..., 3:9]
        joint_rot_obs = sample[..., 9:9 + num_rots * 6]

        root_rot = torch_util.tan_norm_to_quat(root_rot_obs.reshape(-1, 6))
        root_rot = torch_util.quat_to_exp_map(root_rot)
        root_rot = root_rot.reshape(list(root_rot_obs.shape[:-1]) + [3])
        joint_rot = torch_util.tan_norm_to_quat(joint_rot_obs.reshape(list(joint_rot_obs.shape[:-1]) + [num_rots, 6]))
        joint_dof = kin_char_model.rot_to_dof(joint_rot)

        data_frames[..., 0:3] = root_pos_obs
        data_frames[..., 3:6] = root_rot
        data_frames[..., 6:] = joint_dof

        return data_frames

    def _motion_frames_to_init_states(self, motion_frames):
        kin_char_model = self._kin_char_model
        seq_len = motion_frames.shape[1]
        dt = self._engine.get_timestep()

        root_pos = motion_frames[..., 0:3]
        root_rot = torch_util.exp_map_to_quat(motion_frames[..., 3:6])
        dof_pos = motion_frames[..., 6:]
        joint_rot = kin_char_model.dof_to_rot(dof_pos)
        body_pos, body_rot = kin_char_model.forward_kinematics(root_pos, root_rot, joint_rot)

        root_vel = torch.zeros_like(root_pos)
        root_ang_vel = torch.zeros_like(root_pos)
        dof_vel = torch.zeros_like(dof_pos)

        if (seq_len > 1):
            root_vel[:, :-1] = (root_pos[:, 1:] - root_pos[:, :-1]) / dt
            root_vel[:, -1] = root_vel[:, -2]

            drot = torch_util.quat_diff(root_rot[:, :-1], root_rot[:, 1:])
            root_ang_vel[:, :-1] = torch_util.quat_to_exp_map(drot) / dt
            root_ang_vel[:, -1] = root_ang_vel[:, -2]

            dof_dt = torch.full([motion_frames.shape[0], seq_len - 1, 1], dt, device=self._device, dtype=motion_frames.dtype)
            dof_vel_seq = kin_char_model.compute_dof_vel(joint_rot[:, :-1], joint_rot[:, 1:], dof_dt)
            dof_vel[:, :-1] = dof_vel_seq
            dof_vel[:, -1] = dof_vel[:, -2]

        init_states = {
            "root_pos": root_pos,
            "root_rot": root_rot,
            "root_vel": root_vel,
            "root_ang_vel": root_ang_vel,
            "joint_rot": joint_rot,
            "dof_pos": dof_pos,
            "dof_vel": dof_vel,
            "body_pos": body_pos,
            "body_rot": body_rot,
        }

        for k, v in init_states.items():
            init_states[k] = v.unsqueeze(1)
        
        return init_states