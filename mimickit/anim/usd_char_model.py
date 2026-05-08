import numpy as np
import pxr
import torch

import anim.kin_char_model as kin_char_model
import util.torch_util as torch_util

#######################################
## USD Character Model
#######################################

class USDCharModel(kin_char_model.KinCharModel):
    def __init__(self, device):
        super().__init__(device)
        return
    
    def load(self, char_file):
        stage = pxr.Usd.Stage.Open(char_file)
        assert(stage is not None), "Failed to load USD file: {:s}".format(char_file)

        root_prim = self._find_root_prim(stage)
        assert(root_prim is not None), "Failed to find articulation root."

        body_prims, body_names = self._extract_body_prims(stage)
        joint_prims = self._extract_joint_prims(stage)
        num_bodies = len(body_names)

        parent_indices = [None] * num_bodies
        local_translation = [None] * num_bodies
        local_rotation = [None] * num_bodies
        joints = [None] * num_bodies

        parent_indices[0] = -1
        local_translation[0] = np.array([0.0, 0.0, 0.0])
        local_rotation[0] = np.array([0.0, 0.0, 0.0, 1.0])
        joints[0] = self._build_root_joint()

        for joint_prim in joint_prims:
            body_prim0 = joint_prim.GetBody0Rel().GetTargets()
            body_prim1 = joint_prim.GetBody1Rel().GetTargets()

            parent_name = body_prim0[0].name
            child_name = body_prim1[0].name
            parent_idx = body_names.index(parent_name)
            child_idx = body_names.index(child_name)

            pos_data = joint_prim.GetLocalPos0Attr().Get()
            pos = np.array([pos_data[0], pos_data[1], pos_data[2]])
            
            rot0_data = joint_prim.GetLocalRot0Attr().Get()
            rot0 = torch.tensor([[rot0_data.imaginary[0], 
                                 rot0_data.imaginary[1], 
                                 rot0_data.imaginary[2], 
                                 rot0_data.real]], 
                                 device=self._device)
            rot0 = torch_util.quat_normalize(rot0)
            
            rot1_data = joint_prim.GetLocalRot1Attr().Get()
            rot1 = torch.tensor([[rot1_data.imaginary[0], 
                                 rot1_data.imaginary[1], 
                                 rot1_data.imaginary[2], 
                                 rot1_data.real]], 
                                 device=self._device)
            rot1 = torch_util.quat_normalize(rot1)
            
            rot = torch_util.quat_mul(rot0, torch_util.quat_conjugate(rot1))
            rot = rot[0].cpu().numpy()

            curr_joint = self._parse_joint(stage, child_name, joint_prim)

            parent_indices[child_idx] = parent_idx
            local_translation[child_idx] = pos
            local_rotation[child_idx] = rot
            joints[child_idx] = curr_joint

        body_order = self._compute_body_order(parent_indices)

        parent_indices = [body_order.index(i) for i in parent_indices[1:]]
        parent_indices = [-1] + parent_indices
        parent_indices = [parent_indices[i] for i in body_order]
        body_names = [body_names[i] for i in body_order]
        local_translation = [local_translation[i] for i in body_order]
        local_rotation = [local_rotation[i] for i in body_order]
        joints = [joints[i] for i in body_order]

        self.init(body_names=body_names,
                  parent_indices=parent_indices,
                  local_translation=local_translation,
                  local_rotation=local_rotation,
                  joints=joints)
        return
     
    def save(self, output_file):
        assert(False), "USD export is not yet supported."
        return
    
    def _find_root_prim(self, stage):
        root_prim = None

        for prim in stage.Traverse():
            if prim.HasAPI(pxr.UsdPhysics.ArticulationRootAPI):
                root_prim = prim
                break
        
        return root_prim
    
    def _extract_body_prims(self, stage):
        prims = []
        names = []

        for prim in stage.Traverse():
            if prim.HasAPI(pxr.UsdPhysics.RigidBodyAPI):
                body_name = prim.GetPath().name
                prims.append(prim)
                names.append(body_name)

        return prims, names

    def _extract_joint_prims(self, stage):
        prims = []
        for prim in stage.Traverse():
            joint = pxr.UsdPhysics.Joint.Get(stage, prim.GetPath())
            if joint:
                prims.append(joint)
                
        return prims
    
    def _parse_joint(self, stage, body_name, joint_prim):
        prim_path = joint_prim.GetPath()
        
        if (pxr.UsdPhysics.FixedJoint.Get(stage, prim_path)):
            joint = self._parse_fixed_joint(joint_prim)
        elif (pxr.UsdPhysics.RevoluteJoint.Get(stage, prim_path)):
            joint = self._parse_hinge_joint(stage, joint_prim)
        elif (pxr.UsdPhysics.Joint.Get(stage, prim_path)):
            joint = self._parse_sphere_joint(joint_prim)
        else:
            assert(False), "Unsupported joint type found in {:s}.".format(prim_path)
        
        return joint

    def _parse_hinge_joint(self, stage, joint_prim):
        joint_name = joint_prim.GetPath().name
        rev_prim = pxr.UsdPhysics.RevoluteJoint.Get(stage, joint_prim.GetPath())
        rev_axis = rev_prim.GetAxisAttr().Get()

        if (rev_axis == "X"):
            joint_axis = np.array([1.0, 0.0, 0.0])
        elif (rev_axis == "Y"):
            joint_axis = np.array([0.0, 1.0, 0.0])
        elif (rev_axis == "Z"):
            joint_axis = np.array([0.0, 0.0, 1.0])
        else:
            assert(False), "Unsuported hinge joint axis: {:s}".format(rev_axis)

        joint_axis = torch.tensor(joint_axis, device=self._device, dtype=torch.float32)
        joint_axis = joint_axis.unsqueeze(0)

        rot1_data = joint_prim.GetLocalRot1Attr().Get()
        rot1 = torch.tensor([[rot1_data.imaginary[0], 
                              rot1_data.imaginary[1],
                              rot1_data.imaginary[2],
                              rot1_data.real]],
                              device=self._device)
        rot1 = torch_util.quat_normalize(rot1)
        joint_axis = torch_util.quat_rotate(rot1, joint_axis)
        joint_axis = joint_axis[0]

        joint = kin_char_model.Joint(name=joint_name,
                      joint_type=kin_char_model.JointType.HINGE,
                      axis=joint_axis)
        return joint

    def _parse_sphere_joint(self, joint_prim):
        # consolidate series of three hinge joints into a single spherical joint
        joint_name = joint_prim.GetPath().name
        joint = kin_char_model.Joint(name=joint_name,
                                    joint_type=kin_char_model.JointType.SPHERICAL,
                                    axis=None)
        return joint

    def _parse_fixed_joint(self, joint_prim):
        joint_name = joint_prim.GetPath().name
        joint = kin_char_model.Joint(name=joint_name,
                      joint_type=kin_char_model.JointType.FIXED,
                      axis=None)
        return joint
    
    def _compute_body_order(self, parent_indices):
        body_children = self._build_body_children_map(parent_indices)
        
        # recursively adding all bodies into the list in DFS order
        body_order = []
        def _traverse_dfs(body_idx):
            body_order.append(body_idx)
            children_indices = body_children[body_idx]

            for child in children_indices:
                _traverse_dfs(child)
            
        _traverse_dfs(0)

        return body_order