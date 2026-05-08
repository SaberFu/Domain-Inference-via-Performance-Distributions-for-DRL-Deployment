import torch
import sys

sys.path.append(".")

from mimickit.util.torch_util import quat_mul, quat_conjugate, quat_rotate


def compute_global_rotations(local_quats, parents):
    """
    Computes global rotations from local rotations using Forward Kinematics.
    
    Args:
        local_quats: (N, J, 4) torch tensor of quaternions (x, y, z, w).
        parents: List of length J, where parents[i] is the parent index of joint i.
                 -1 indicates the root. Must be topologically sorted.
                 
    Returns:
        global_quats: (N, J, 4) torch tensor of global quaternions.
    """
    N, J, _ = local_quats.shape
    
    globals_per_joint = [None] * J
    
    for i, parent_idx in enumerate(parents):
        if parent_idx == -1:
            globals_per_joint[i] = local_quats[:, i, :]
        else:
            globals_per_joint[i] = quat_mul(globals_per_joint[parent_idx], local_quats[:, i, :])
            
    global_quats = torch.stack(globals_per_joint, dim=1)
    
    return global_quats

def compute_local_rotations(global_quats, parents):
    """
    Computes local rotations from global rotations using Inverse Kinematics algebra.
    
    Args:
        global_quats: (N, J, 4) torch tensor.
        parents: List of length J.
        
    Returns:
        local_quats: (N, J, 4) torch tensor.
    """
    N, J, _ = global_quats.shape
    
    locals_per_joint = [None] * J
    
    for i, parent_idx in enumerate(parents):
        if parent_idx == -1:
            locals_per_joint[i] = global_quats[:, i, :]
        else:
            parent_q = global_quats[:, parent_idx, :]
            child_q = global_quats[:, i, :]
            
            parent_inv = quat_conjugate(parent_q)
            locals_per_joint[i] = quat_mul(parent_inv, child_q)
            
    local_quats = torch.stack(locals_per_joint, dim=1)
    
    return local_quats

def compute_global_translations(global_rotations, local_offsets, parents):
    """
    Computes global translations (positions) given global rotations and local structural offsets.

    Args:
        global_rotations: (N, J, 4) torch tensor of global quaternions (x, y, z, w).
                          These are the 'new' rotations you calculated previously.
        local_offsets: (J, 3) torch tensor.
                       The local translation of each joint relative to its parent 
                       (often called bone offsets or bind pose translations).
        parents: List of length J with parent indices (-1 for root).

    Returns:
        global_translations: (N, J, 3) torch tensor of global positions.
    """
    N, J, _ = global_rotations.shape
    
    global_translations = torch.zeros((N, J, 3), dtype=global_rotations.dtype, device=global_rotations.device)
    
    for i, parent_idx in enumerate(parents):
        if parent_idx == -1:
            global_translations[:, i, :] = local_offsets[i]
        else:
            parent_pos = global_translations[:, parent_idx, :]
            parent_rot = global_rotations[:, parent_idx, :]
            
            offset = local_offsets[i].unsqueeze(0).expand(N, -1)
            offset_rotated = quat_rotate(parent_rot, offset)
            
            global_translations[:, i, :] = parent_pos + offset_rotated
            
    return global_translations