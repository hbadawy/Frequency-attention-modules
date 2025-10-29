
import torch
import torch.nn as nn
from scipy.fftpack import dct, idct

class BalancedFrequencyAttention(nn.Module):
    def __init__(self, input_dim, low_freq_weight=0.6, high_freq_weight=0.4):
        super(BalancedFrequencyAttention, self).__init__()
        self.low_freq_weight = low_freq_weight
        self.high_freq_weight = high_freq_weight
        self.fc1 = nn.Linear(input_dim, input_dim // 4, bias=False)
        self.fc2 = nn.Linear(input_dim // 4, input_dim, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply DCT on the input along spatial dimensions
        batch_size, channels, height, width = x.shape
        x_reshaped = x.cpu().numpy() #.view(batch_size, channels, -1).cpu().numpy()  # Flatten spatial dimensions
        print (x_reshaped.shape)  # Debugging line to check the shape after reshaping
        x_dct = dct(x_reshaped, axis=-1, norm='ortho')
        print ("x_dct",x_dct.shape)  # Debugging line to check the shape after DCT
        
        # Split into low and high frequencies
        split_point1 = x_dct.shape[-1] // 2
        split_point2 = x_dct.shape[-1] // 4
        print (split_point1, split_point2)
        low_freq = x_dct[:, :, :split_point1, :split_point1]
        high_freq = x_dct[:, :, height-split_point2:, height-split_point2:]
        print (low_freq.shape, high_freq.shape)
        
        high_freq = high_freq.reshape(batch_size, channels, low_freq.shape[2], low_freq.shape[3])
        # Balance frequencies using weighted sum
        balanced_freq = (self.low_freq_weight * low_freq + self.high_freq_weight * high_freq)
        print (balanced_freq.shape)
        
        # Transform back to spatial domain using inverse DCT
        x_balanced = idct(balanced_freq, axis=-1, norm='ortho')
        print (x_balanced.shape)
        x_balanced = torch.tensor(x_balanced, device=x.device) #view(batch_size, channels, height, width)
        
        # Global Average Pooling
        gap = torch.mean(x_balanced, dim=(2, 3))
        
        # Fully Connected layers to generate attention weights
        attention = self.fc1(gap)
        attention = self.fc2(attention)
        attention = self.sigmoid(attention)
        
        # Apply attention weights to input features
        attention = attention.view(batch_size, channels, 1, 1)
        output = x * attention
        
        return output

if __name__ == "__main__":
    # Example usage

    input_tensor = torch.rand(4, 64, 32, 32)  # Batch of 4, 64 channels, 32x32 spatial dimensions
    freq_attention = BalancedFrequencyAttention(input_dim=64)
    output_tensor = freq_attention(input_tensor)
    print (output_tensor.shape)  # Should be the same shape as input_tensor (4, 64, 32, 32)
