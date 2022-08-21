import torch


class MLPNet(torch.nn.Module):
    def __init__(self, input_dims, output_dims) -> None:
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_dims, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, self.output_dims),
        )
        
    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
            
        logits = self.model(obs.view(obs.shape[0], -1))
        return logits, state
