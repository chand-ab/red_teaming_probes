from trl import SFTTrainer
import torch

## custom SFTTRAINER
class customSFTTrainer(SFTTrainer):
    def __init__(self, layer, scaler, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer = layer
        self.scaler = scaler
        self.activations = None
    
    def compute_loss(self,*args, **kwargs):
        def hook(module, input, output):
            self.activations = output[0] if isinstance(output, tuple) else output
        # get layer module
        submodule = self.model.get_submodule(self.layer)
        # register_forward_hook
        handle = submodule.register_forward_hook(hook)
        default_loss = super().compute_loss(*args, **kwargs)
        handle.remove()
        # need to project onto the probe direction
        custom_loss = (default_loss+ 
                       self.scaler * torch.linalg.norm(self.activations, p=2, dim=-1).mean())
        return custom_loss

## probe direction
