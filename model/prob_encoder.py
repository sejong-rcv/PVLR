import torch
import torch.nn as nn

class SnippetEncoder(nn.Module): 
    """
    Build a PSRL network with: Beyesian dropout inference encoder
    """
    def __init__(self, dim=2048, prob=False, mu_size=None):
        super(SnippetEncoder, self).__init__()
        self.prob = prob
        self.dim = dim
        if self.prob == 1:
            self.fc_mean = nn.Linear(2048, 1024)
            self.fc_var = nn.Linear(2048, 1024)
            self.relu = nn.ReLU()
            self.layer_norm = nn.LayerNorm(dim)
            self.dropout = nn.Dropout(0.2)
            self.sigmoid = nn.Sigmoid()

    def sample_gaussian_tensors(self, mu, logsigma, num_samples):
        '''
        Reprameterization trick for sampling
        '''
        eps = torch.randn(num_samples, mu.size(0), mu.size(1), mu.size(2), dtype=mu.dtype, device=mu.device, requires_grad=True)
        samples = torch.einsum('nbtd,btd->nbtd', [eps, torch.exp(.5*logsigma)]) 
        samples = samples.permute(1,0,2,3)
        samples = samples + mu.unsqueeze(1).repeat(1,num_samples,1,1)
        return samples 

    def forward(self, itr, f, clip_feat, num_samples): 
        if self.prob == 1:
            f = f.permute((0, 2, 1))
            mu = self.fc_mean(f)
            var = self.relu(self.fc_var(f))
            embedding = self.sample_gaussian_tensors(mu, var, num_samples)
            return mu, embedding, var 
        else:
            return f, f, f
        

class TextEncoder(nn.Module):
    """
    Build a PSRL network with: Beyesian dropout inference encoder
    """
    def __init__(self, dim = 2048, prob = False, mu_size=None):
        """
        dim = feature dimension (default: 512)
        """
        super(TextEncoder, self).__init__()

        self.prob = prob
        self.dim = dim

        if self.prob == 1:
            self.fc_mean = nn.Linear(self.dim, self.dim)
            self.fc_var = nn.Linear(self.dim, self.dim)
            self.relu = nn.ReLU()
            self.layer_norm = nn.LayerNorm(dim)
            self.dropout = nn.Dropout(0.2)
            self.sigmoid = nn.Sigmoid()

    def sample_gaussian_tensors(self, mu, logsigma, num_samples):
        '''
        Reprameterization trick for sampling
        '''
        eps = torch.randn(num_samples, mu.size(0), mu.size(1), dtype=mu.dtype, device=mu.device, requires_grad=True)
        samples = torch.einsum('mcd,cd->mcd', [eps, torch.exp(.5*logsigma)])  
        samples = samples + mu.unsqueeze(0).repeat(num_samples,1,1)

        return samples # [5, 21, 2048]

    def forward(self, itr, f, num_samples):
        if self.prob == 1:
            mu = f
            var = self.fc_var(f)
            embedding = self.sample_gaussian_tensors(mu, var, num_samples)
            return mu, embedding, var 
        else:
            return f, f, f