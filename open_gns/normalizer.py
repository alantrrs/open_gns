
# Normalizer
import torch

class Normalizer():
    def __init__(self, size, mask_cols=None, device=None):
        self.x_sum = torch.zeros(size)
        self.x_centered_sum2 = torch.zeros(size)
        if device is not None:
            self.x_sum = self.x_sum.to(device)
            self.x_centered_sum2 = self.x_centered_sum2.to(device)
        self.num_particles = 0
        self.mask_cols = mask_cols

    def __call__(self, x):
        self.num_particles += x.size(0)
        self.x_sum += torch.sum(x,0)
        mean = torch.div(self.x_sum, self.num_particles)
        zero_mean = x - mean
        self.x_centered_sum2 += torch.sum(torch.pow(zero_mean, 2), 0)
        var = torch.div(self.x_centered_sum2, self.num_particles) + 1.0e-10
        normalized_x = torch.div(zero_mean, var)
        if self.mask_cols is not None:
            normalized_x[:,self.mask_cols] = x[:, self.mask_cols]
        return normalized_x
