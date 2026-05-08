import enum
import numpy as np

class CameraMode(enum.Enum):
    still = 0
    track = 1

class Camera:
    def __init__(self, mode, engine, pos, target, track_env_id=0, track_obj_id=0):
        self._mode = mode
        self._engine = engine
        self._track_env_id = track_env_id
        self._track_obj_id = track_obj_id
        self._pos = pos.copy()
        self._target = target.copy()
        
        self.lookat(pos, target)
        return

    def update(self):
        if (self._mode is CameraMode.still):
            pass
        elif (self._mode is CameraMode.track):
            cam_pos = self._engine.get_camera_pos()
            cam_dir = self._engine.get_camera_dir()

            prev_pos = self._pos
            prev_target = self._target
            prev_delta = prev_target - prev_pos
            prev_dist = np.linalg.norm(prev_delta)
            prev_dir = prev_delta / prev_dist

            if (not np.allclose(prev_pos, cam_pos, atol=1e-5)):
                cam_delta = cam_pos - prev_target
            elif (not np.allclose(prev_dir, cam_dir, atol=1e-5)):
                cam_delta = -prev_dist * cam_dir
            else:
                cam_delta = cam_pos - prev_target

            obj_pos = self._engine.get_root_pos(self._track_obj_id)
            obj_pos = obj_pos[self._track_env_id].cpu().numpy()

            new_target = np.array([obj_pos[0], obj_pos[1], prev_target[2]])
            new_pos = new_target + cam_delta
            
            self.lookat(new_pos, new_target)
        else:
            assert(False), "Unsupported camera mode {}".format(self._mode)
        return

    def lookat(self, pos, target):
        self._engine.set_camera_pose(pos, target)
        self._pos[:] = pos
        self._target[:] = target
        return
