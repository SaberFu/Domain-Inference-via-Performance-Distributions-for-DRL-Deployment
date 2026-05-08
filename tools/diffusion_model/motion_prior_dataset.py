import sys
sys.path.append(".")
sys.path.append("mimickit")

import gymnasium.spaces as spaces
import torch
import numpy as np

import pickle
import yaml
import os

import util.torch_util as torch_util
from envs.amp_env import compute_disc_obs
import anim.motion_lib as motion_lib
from tools.util.char_vis_util import output_body_pos_anim

class MotionPriorData:
    def __init__(self, config):
        with open(config["env_config"], "r") as stream:
            env_config = yaml.safe_load(stream)
        
        self._device = config["device"]
        self._global_obs = env_config["global_obs"]
        self._root_height_obs = env_config.get("root_height_obs", True)

        self._num_disc_obs_steps = env_config["num_disc_obs_steps"]
        self._enable_tar_obs = env_config.get("enable_tar_obs", False)
        self.control_freq = config["control_freq"]
        self._timestep = 1.0 / self.control_freq
        self._disc_dof_vel_obs = env_config.get("disc_dof_vel_obs", False)

        self._build_envs(config)

        key_bodies = env_config.get("key_bodies", [])
        self._key_body_ids = self._build_body_ids_tensor(key_bodies)
        return

    def _has_key_bodies(self):
        return len(self._key_body_ids) > 0
    
    def _track_global_root(self):
        return self._enable_tar_obs and self._global_obs
    
    def _build_body_ids_tensor(self, body_names):
        assert isinstance(body_names, list)
        body_ids = []
        for body_name in body_names:
            body_id = self._kin_char_model.get_body_id(body_name)
            body_ids.append(body_id)
        return body_ids

    def _build_envs(self, config):
        self._load_char_asset(config)
        motion_file = config["motion_file"]
        self._load_motions(motion_file)
        return
    
    def _load_motions(self, motion_file):
        self._motion_lib = motion_lib.MotionLib(motion_file=motion_file, 
                                                kin_char_model=self._kin_char_model,
                                                device=self._device)
        return
        
    
    def _load_char_asset(self, config):
        with open(config["env_config"], "r") as stream:
            env_config = yaml.safe_load(stream)
        
        char_file = env_config["char_file"]
        self._build_kin_char_model(char_file)
        return

    def _build_kin_char_model(self, char_file):
        _, file_ext = os.path.splitext(char_file)
        if (file_ext == ".xml"):
            import anim.mjcf_char_model as mjcf_char_model
            char_model = mjcf_char_model.MJCFCharModel(self._device)
        elif (file_ext == ".urdf"):
            import anim.urdf_char_model as urdf_char_model
            char_model = urdf_char_model.URDFCharModel(self._device)
        elif (file_ext == ".usd"):
            import anim.usd_char_model as usd_char_model
            char_model = usd_char_model.USDCharModel(self._device)
        else:
            print("Unsupported character file format: {:s}".format(file_ext))
            assert(False)

        self._kin_char_model = char_model
        self._kin_char_model.load(char_file)

        return
    
    def fetch_obs_demo(self, num_samples, return_class=False):
        obs = self.fetch_smp_obs_demo(num_samples, return_id=return_class)
        return obs
            

    @torch.no_grad()
    def fetch_smp_obs_demo(self, num_samples, return_id=False):
        motion_ids = self._motion_lib.sample_motions(num_samples)
        motion_times0 = self._motion_lib.sample_time(motion_ids)
        smp_obs = self._compute_smp_obs_demo(motion_ids, motion_times0)
        return smp_obs
    
    def _compute_smp_obs_demo(self, motion_ids, motion_times0):
        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, body_pos = self._fetch_smp_demo_data(motion_ids, motion_times0)
        
        if (self._track_global_root()):
            ref_root_pos = torch.zeros_like(root_pos[..., -1, :])
            ref_root_rot = torch.zeros_like(root_rot[..., -1, :])
            ref_root_rot[..., -1] = 1
        else:
            ref_root_pos = root_pos[..., -1, :]
            ref_root_rot = root_rot[..., -1, :]
        
        if (self._has_key_bodies()):
            key_pos = body_pos[..., self._key_body_ids, :]
        else:
            key_pos = torch.zeros([0], device=self._device)

        smp_obs = compute_disc_obs(ref_root_pos=ref_root_pos,
                                  ref_root_rot=ref_root_rot,
                                  root_pos=root_pos,
                                  root_rot=root_rot, 
                                  root_vel=root_vel,
                                  root_ang_vel=root_ang_vel,
                                  joint_rot=joint_rot,
                                  dof_vel=dof_vel,
                                  key_pos=key_pos,
                                  global_obs=self._global_obs,
                                  root_height_obs=self._root_height_obs,
                                  dof_vel_obs=self._disc_dof_vel_obs)
        
        return smp_obs

    def _fetch_smp_demo_data(self, motion_ids, motion_times0):
        num_samples = motion_ids.shape[0]

        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self._num_disc_obs_steps])
        motion_times = motion_times0.unsqueeze(-1)
        time_steps = -self._timestep * torch.arange(0, self._num_disc_obs_steps, device=self._device)
        time_steps = torch.flip(time_steps, dims=[0])
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)
        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel = self._motion_lib.calc_motion_frame(motion_ids, motion_times)
        
        body_pos, _ = self._kin_char_model.forward_kinematics(root_pos, root_rot, joint_rot)

        root_pos = torch.reshape(root_pos, shape=[num_samples, self._num_disc_obs_steps, root_pos.shape[-1]])
        root_rot = torch.reshape(root_rot, shape=[num_samples, self._num_disc_obs_steps, root_rot.shape[-1]])
        root_vel = torch.reshape(root_vel, shape=[num_samples, self._num_disc_obs_steps, root_vel.shape[-1]])
        root_ang_vel = torch.reshape(root_ang_vel, shape=[num_samples, self._num_disc_obs_steps, root_ang_vel.shape[-1]])
        joint_rot = torch.reshape(joint_rot, shape=[num_samples, self._num_disc_obs_steps, joint_rot.shape[-2], joint_rot.shape[-1]])
        dof_vel = torch.reshape(dof_vel, shape=[num_samples, self._num_disc_obs_steps, dof_vel.shape[-1]])
        body_pos = torch.reshape(body_pos, shape=[num_samples, self._num_disc_obs_steps, body_pos.shape[-2], body_pos.shape[-1]])
        
        return root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, body_pos
    
    def get_obs_space(self):
        smp_obs = self.fetch_obs_demo(1)
        smp_obs_shape = list(smp_obs.shape[1:])
        smp_obs_dtype = torch_util.torch_dtype_to_numpy(smp_obs.dtype)
        smp_obs_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=smp_obs_shape,
            dtype=smp_obs_dtype,
        )

        return smp_obs_space

    def plot_jnt(self, jnt_pos, out_path=None):
        parent_indices = self._motion_lib._kin_char_model._parent_indices
        jnt_pos = jnt_pos.squeeze()
        
        if len(jnt_pos.shape)==4:
            num_char = jnt_pos.shape[0]
            num_frame = jnt_pos.shape[1]
            num_jnt = jnt_pos.shape[2]
            jnt_pos = jnt_pos.permute(1,0,2,3).reshape(num_frame, -1, jnt_pos.shape[3])
            parent_indices = [
                parent_idx + char_idx * num_jnt if parent_idx >= 0 else -1
                for char_idx in range(num_char)
                for parent_idx in parent_indices
            ]
        elif len(jnt_pos.shape)==3:
            pass
        elif len(jnt_pos.shape)==2:
            jnt_pos = torch.cat([jnt_pos[None,...] for _ in range(5)],dim=0)
        else:
            raise NotImplementedError("wrong data dim for plot")

        jnt_pos = jnt_pos.cpu().detach().numpy()
        output_body_pos_anim(jnt_pos, parent_indices, save_path=out_path, fps=5)
        return
    
    def convert_sample_to_frames(self, sample):
        return self.smp_obs_to_motion_frames(sample)
        
    def smp_obs_to_motion_frames(self, sample):
        num_joints = self._motion_lib.get_num_joints() - 1
        frame_dim = self._motion_lib.get_motion_frame_size()
        data_frames = torch.zeros(sample.shape[0], frame_dim)

        root_pos_obs = sample[..., 0: 3]
        root_rot_obs = sample[..., 3: 9]
        joint_rot_obs = sample[..., 9: 9 + num_joints * 6]

        root_pos = root_pos_obs
        root_rot = torch_util.tan_norm_to_quat(root_rot_obs)
        root_rot = torch_util.quat_to_exp_map(root_rot)
        joint_rot = torch_util.tan_norm_to_quat(joint_rot_obs.reshape(list(joint_rot_obs.shape[: 1]) + [num_joints, 6]))

        joint_dof = self._motion_lib._kin_char_model.rot_to_dof(joint_rot)

        data_frames[:, 0: 3] = root_pos
        data_frames[:, 3: 6] = root_rot
        data_frames[:, 6:] = joint_dof

        return data_frames
    
    def calc_joint_position_from_frame(self, data_frames):
        root_pos, root_rot, joint_rot = self._motion_lib._extract_frame_data(data_frames)
        body_pos, body_rot = self._motion_lib._kin_char_model.forward_kinematics(root_pos, root_rot, joint_rot)
        return body_pos
