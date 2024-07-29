import torch
import torch.nn as nn

class SnippetEncoder(nn.Module): 
    """
    Build a PSRL network with: Beyesian dropout inference encoder
    """
    def __init__(self, dim=2048, eps_std=0.01, mu_size=None):
        """
        dim = feature dimension (default: 512)
        """
        super(SnippetEncoder, self).__init__()
        self.dim = dim
        self.fc_mean = nn.Linear(2048, 1024)
        self.fc_var = nn.Linear(2048, 1024)
        self.relu = nn.ReLU()
        self.eps_std = eps_std

    def sample_gaussian_tensors(self, mu, logsigma, num_samples):
        '''
        Reprameterization trick for sampling
        '''
        # eps = torch.randn(num_samples, mu.size(0), mu.size(1), mu.size(2), dtype=mu.dtype, device=mu.device, requires_grad=True)
        eps = torch.normal(0, self.eps_std, size=(num_samples, mu.size(0), mu.size(1), mu.size(2)))
        samples = torch.einsum('nbtd,btd->nbtd', [eps, torch.exp(.5*logsigma)]) 
        samples = samples.permute(1,0,2,3)
        samples = samples + mu.unsqueeze(1).repeat(1,num_samples,1,1)
        return samples # [10, 5, 320, 2048]

    def forward(self, itr, f, num_samples, split): # flow: [10, 320, 1024]
        f = f.permute((0, 2, 1))
        mu = self.fc_mean(f)
        var = self.relu(self.fc_var(f))
        if split == 'train':
            embedding = self.sample_gaussian_tensors(mu, var, num_samples)
            return mu, embedding, var 
        else:
            return mu, mu.unsqueeze(1), var 
    