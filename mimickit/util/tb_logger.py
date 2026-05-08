import numbers
import numpy as np
import os
import tensorboardX

import util.logger as logger
import util.video as video

class TBLogger(logger.Logger):
    MISC_TAG = "Misc"

    def __init__(self):
        super().__init__()

        self._writer = None
        self._step_var_key = None
        self._collections = dict()
        return
    
    def reset(self):
        super().reset()
        return

    def configure_output_file(self, filename=None):
        super().configure_output_file(filename)

        if (logger.Logger.is_root()):
            output_dir = os.path.dirname(filename)
            self._writer = tensorboardX.SummaryWriter(output_dir)
            
        return

    def log(self, key, val, collection=None, quiet=False):
        super().log(key, val, collection, quiet)

        if (collection is not None):
            self._add_collection(collection, key)
        return

    def write_log(self):
        row_count = self._row_count

        super().write_log()

        if (logger.Logger.is_root() and (self._writer is not None)):
            if (row_count == 0):
                self._key_tags = self._build_key_tags()
            
            step_val = row_count
            if (self._step_key is not None):
                step_val = self.log_current_row.get(self._step_key, "").val
            
            for i, key in enumerate(self.log_headers):
                if (key != self._step_key):
                    entry = self.log_current_row.get(key, "")
                    val = entry.val
                    tag = self._key_tags[i]

                    if (isinstance(val, video.Video)):
                        self._write_video(tag, val, step_val)
                    elif (isinstance(entry.val, numbers.Number)):
                        self._writer.add_scalar(tag, val, step_val)
                    else:
                        assert(False), "Unsupported Tensorboard value type: {}".format(type(val))
        return
    
    def _add_collection(self, name, key):
        if (name not in self._collections):
            self._collections[name] = []
        self._collections[name].append(key)
        return
    
    def _build_key_tags(self):
        tags = []
        for key in self.log_headers:
            curr_tag = TBLogger.MISC_TAG
            for col_tag, col_keys in self._collections.items():
                if key in col_keys:
                    curr_tag = col_tag

            curr_tags = "{:s}/{:s}".format(curr_tag, key)
            tags.append(curr_tags)

        return tags
    
    def _write_video(self, tag, video, step_val):
        fps = video.get_fps()
        frames = video.get_frames()
        frames = np.array(frames)
        frames = frames.transpose(0, 3, 1, 2)
        frames = np.expand_dims(frames, axis=0)

        self._writer.add_video(tag, frames, step_val, fps)
        return