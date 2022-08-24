import torch


class MLPNet(torch.nn.Module):
    def __init__(self, input_dims, output_dims, device) -> None:
        super().__init__()
        self.device = device
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
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
            
        logits = self.model(obs.view(obs.shape[0], -1))
        return logits, state
    

class CNNNet(torch.nn.Module):
    def __init__(self, input_height, input_width, input_channels, num_actions, device):
        super().__init__()
        self.device = device
        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels
        self.num_actions = num_actions
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(self.input_channels, 32, kernel_size=8, stride=4, padding=0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        )
        
        with torch.no_grad():
            random_noise = torch.rand(1, self.input_channels, self.input_height, self.input_width)
            n_flatten = self.cnn(random_noise).shape[1]
            
        self.q_estimate = torch.nn.Sequential(
            torch.nn.Linear(n_flatten, self.num_actions),
            torch.nn.ReLU()
        )
        
    def forward(self, obs, state=None, info={}):
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        obs = obs.permute(0, 3, 1, 2)
        obs = self.cnn(obs)
        obs = self.q_estimate(obs)
        return obs, state
