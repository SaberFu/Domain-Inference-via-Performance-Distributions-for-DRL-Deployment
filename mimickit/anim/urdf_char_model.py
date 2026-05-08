import numpy as np
import torch
import xml.etree.ElementTree as ET

import anim.kin_char_model as kin_char_model
import util.torch_util as torch_util

#######################################
## URDF Character Model
#######################################

class URDFCharModel(kin_char_model.KinCharModel):
    def __init__(self, device):
        super().__init__(device)
        return
    
    def load(self, char_file):
        tree = ET.parse(char_file)
        xml_root = tree.getroot()
        
        body_names = self._parse_body_list(xml_root)
        num_bodies = len(body_names)

        parent_indices = [-1] * num_bodies
        local_translation = [None] * num_bodies
        local_rotation = [None] * num_bodies
        joints = [None] * num_bodies

        self._body_names = body_names
        body_id_map = self._build_name_body_map()

        root_joint = self._build_root_joint()
        local_translation[0] = np.array([0.0, 0.0, 0.0])
        local_rotation[0] = np.array([0.0, 0.0, 0.0, 1.0])
        joints[0] = root_joint

        joints_data = xml_root.findall("joint")
        
        for curr_joint_data in joints_data:
            parent_name = curr_joint_data.find("parent").attrib.get("link")
            child_name = curr_joint_data.find("child").attrib.get("link")
            child_id = body_id_map[child_name]
            parent_id = body_id_map[parent_name]
            assert(parent_id < child_id)

            origin = curr_joint_data.find("origin")
            pos = np.array([0.0, 0.0, 0.0])
            rot = np.array([0.0, 0.0, 0.0])

            if (origin is not None):
                pos_data = origin.attrib.get("xyz")
                rot_data = origin.attrib.get("rpy")

                if (pos_data is not None):
                    pos = np.fromstring(pos_data, dtype=float, sep=" ")

                if (rot_data is not None):
                    rot = np.fromstring(rot_data, dtype=float, sep=" ")

            local_translation[child_id] = pos

            rot = torch.tensor(rot)
            quat = torch_util.euler_xyz_to_quat(rot[0], rot[1], rot[2])
            quat = torch_util.quat_normalize(quat)
            local_rotation[child_id] = quat.cpu().numpy()

            joint = self._parse_joint(curr_joint_data)
            joints[child_id] = joint
            parent_indices[child_id] = parent_id

        self.init(body_names=body_names,
                  parent_indices=parent_indices,
                  local_translation=local_translation,
                  local_rotation=local_rotation,
                  joints=joints)
        return
    
    def save(self, output_file):
        assert(False), "URDF export is not yet supported."
        return
    
    def _parse_body_list(self, xml_root):
        body_names =[]

        links_data = xml_root.findall("link")
        joints_data = xml_root.findall("joint")

        children_link_names = []
        for joint in joints_data:
            child_link = joint.find("child").attrib.get("link")
            children_link_names.append(child_link)

        root_name = None
        for link in links_data:
            link_name = link.attrib.get("name")
            if (link_name not in children_link_names):
                root_name = link_name
                break
        assert(root_name is not None)

        # recursively adding all links into the list in DFS order
        def _add_xml_link(link_name):
            body_names.append(link_name)

            for joint in joints_data:
                parent_link = joint.find("parent").attrib.get("link")
                if (parent_link == link_name):
                    child_name = joint.find("child").attrib.get("link")
                    _add_xml_link(child_name)
            
        _add_xml_link(root_name)
        
        return body_names
    
    def _parse_joint(self, xml_joint_data):
        joint_type_str = xml_joint_data.attrib.get("type")
        if (joint_type_str == "revolute"):
            joint = self._parse_revolute_joint(xml_joint_data)
        elif (joint_type_str == "fixed"):
            joint = self._parse_fixed_joint(xml_joint_data)
        else:
            assert(False), "Unsupported joint type: {:s}".format(joint_type_str)
        
        return joint

    def _parse_revolute_joint(self, xml_joint_data):
        joint_name = xml_joint_data.attrib.get("name")
        
        axis_data = xml_joint_data.find("axis")
        joint_axis = np.fromstring(axis_data.attrib.get("xyz"), dtype=float, sep=" ")
        joint_axis = torch.tensor(joint_axis, device=self._device, dtype=torch.float32)

        joint = kin_char_model.Joint(name=joint_name,
                                    joint_type=kin_char_model.JointType.HINGE,
                                    axis=joint_axis)
        return joint

    def _parse_fixed_joint(self, xml_joint_data):
        joint_name = xml_joint_data.attrib.get("name")
        joint = kin_char_model.Joint(name=joint_name,
                                     joint_type=kin_char_model.JointType.FIXED,
                                     axis=None)
        return joint