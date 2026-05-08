import torch
import torch.nn.functional as F
import torch.nn as nn

class ClassifierFreeSampleModel(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model  # model is the actual model to run

    def forward(self, x, timestep, class_labels=None, cfg_scale=1., **kwargs):
        bs = x.shape[0]
        force_drop_ids = torch.concat((torch.zeros((bs), device=x.device, dtype=torch.bool), torch.ones((bs), device=x.device, dtype=torch.bool)), dim=0)
        x_dummy = torch.cat((x, x), dim=0)
        timestep_dummy = torch.cat((timestep, timestep), dim=0)

        if class_labels is not None:
            class_labels_dummy = torch.cat((class_labels, class_labels), dim=0)
        else:
            class_labels_dummy = None

        kwargs["force_drop_ids"] = force_drop_ids
        out = self.model(x_dummy, timestep_dummy, class_labels=class_labels_dummy, **kwargs)
        cond_pred, uncond_pred = out.chunk(2, dim=0) 
        out = uncond_pred + (cfg_scale * (cond_pred - uncond_pred))

        return out
