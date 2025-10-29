
import torch
import torch.nn as nn
from torchvision.models import vgg16
import numpy as np
import torch.nn.functional as F
import torch.fft  # For DCT, we use Fourier Transform equivalents


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio = 8, device=None):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels,in_channels//ratio,1,bias=False, device=device)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels//ratio, in_channels,1,bias=False, device=device)
        self.sigmod = nn.Sigmoid()

    def forward(self,x):
        # avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        avg_out = (self.avg_pool(x))
        # print ("avg out :  ", avg_out.shape)
        avg_out = (self.fc1(avg_out))
        # print ("avg out :  ", avg_out.shape)
        avg_out = self.fc2(self.relu1(avg_out))


        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmod(out)




# class SpatialAttention(nn.Module):
#     def __init__(self, device=None):
#         super(SpatialAttention,self).__init__()

#         self.conv1 = nn.Conv2d(2,1,7,padding=3,bias=False, device=device)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = torch.mean(x,dim=1,keepdim=True)
#         # print (avg_out.shape)
#         max_out = torch.max(x,dim=1,keepdim=True,out=None)[0]
#         # print (max_out.shape)

#         x = torch.cat([avg_out,max_out],dim=1)
#         x = self.conv1(x)

#         return self.sigmoid(x)
    


class My_SpatialAttention(nn.Module):
    def __init__(self, device=None):
        super(My_SpatialAttention,self).__init__()

        self.conv1 = nn.Conv2d(2,1,7,padding=3,bias=False, device=device)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, guiding_map0):

        guiding_map0 = F.interpolate(guiding_map0, x.size()[2:], mode='bilinear', align_corners=True)
        # print ("guiding_map0", guiding_map0.shape)
        guiding_map = F.sigmoid(guiding_map0)
        # print ("guid map max:   ", guiding_map.max(), "guid map min:   ", guiding_map.min())


        avg_out = torch.mean(x,dim=1,keepdim=True)
        # print (avg_out.shape)
        max_out = torch.max(x,dim=1,keepdim=True,out=None)[0]
        # print (max_out.shape)

        avg_out_guided = avg_out * guiding_map
        max_out_guided = max_out * guiding_map

        # x = torch.cat([avg_out,max_out],dim=1)
        x = torch.cat([avg_out_guided,max_out_guided],dim=1)

        x = self.conv1(x)
        
        return self.sigmoid(x)
    




class FrequencyAttention(nn.Module):       #source: copilot!
    """
    Frequency-based Attention Module using DCT
    """
    def __init__(self, in_channels):
        super(FrequencyAttention, self).__init__()
        self.in_channels = in_channels

        # Learnable weights for high and low-frequency components
        self.high_freq_weight = nn.Parameter(torch.ones(in_channels, 1, 1))
        self.low_freq_weight = nn.Parameter(torch.ones(in_channels, 1, 1))
        self.alpha1 = 1
        self.alpha2 = 0

        # A simple activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch_size, channels, height, width)
        b, c, h, w = x.size()

        # Apply Discrete Cosine Transform (DCT)
        dct = torch.fft.fft2(x, norm="ortho").real  # Use FFT for DCT equivalent     # [1, 8, 256, 256]
        # print ("dct shape:   ", dct.shape)   

        # Split into low-frequency and high-frequency components
        low_freq = dct[:, :, :h//2, :w//2]  # Top-left corner for low frequencies          #[1, 8, 128, 128]
        high_freq = dct[:, :, h//2:, w//2:]  # Bottom-right corner for high frequencies
        # print ("low_freq:   ", low_freq.shape)

        # Aggregate frequency components (mean pooling here)
        low_freq_mean = low_freq.mean(dim=(2, 3), keepdim=True)                  #[1, 8, 1, 1]
        high_freq_mean = high_freq.mean(dim=(2, 3), keepdim=True)
        # print ("low_freq_mean:   ", low_freq_mean.shape)

        # Compute attention weights
        low_weighted = low_freq_mean * self.low_freq_weight
        high_weighted = high_freq_mean * self.high_freq_weight
        # print ("self.low_freq_weight:     ", self.low_freq_weight.shape)    #[8, 1, 1]
        # print ("low weight:   ", low_weighted.shape)   #[1, 8, 1, 1]


        # Combine attention weights
        attention = self.sigmoid((self.alpha1 *low_weighted) + (self.alpha2 *high_weighted))
        # print ("attention size:   ",  attention.shape)   #[1, 8, 1, 1]
        # print ("x shape:   ", x.shape)

        # Apply attention to the input feature map
        output = x * attention

        return output
    



class FrequencyAttention2(nn.Module):     # SAM+CAM+FAM2+conv
    def __init__(self, in_channels):
        super(FrequencyAttention2, self).__init__()
        # Learnable weights to emphasize or suppress specific frequency ranges
        self.attention_weights = nn.Parameter(torch.ones(in_channels, dtype=torch.float32))
        self.output = nn.Conv2d(in_channels, in_channels, kernel_size=1)   # I added this line
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch_size, in_channels, height, width)
        batch_size, in_channels, height, width = x.size()

        # Apply 2D Fast Fourier Transform (FFT) to each channel
        freq_domain = torch.fft.fft2(x, dim=(-2, -1))
        freq_magnitude = torch.abs(freq_domain)  # Get the magnitude of frequencies
        
        # Apply attention weights to the frequency magnitude
        refined_freq = freq_magnitude * self.attention_weights.view(1, -1, 1, 1)

        # Reconstruct the feature maps from the modified frequency domain
        freq_domain_refined = torch.fft.ifft2(refined_freq * torch.exp(1j * torch.angle(freq_domain)), dim=(-2, -1))
        refined_features = freq_domain_refined.real  # Take the real part
        
        refined_features = self.output(refined_features)     # I added this line

        refined_features = self.sigmoid(refined_features)

        return refined_features





class FrequencySelfAttention3(nn.Module):   #SAM+CAM+FAM3+selfattn
    def __init__(self, channels):
        super(FrequencySelfAttention3, self).__init__()
        # Key, Query, and Value projections
        self.query = nn.Conv2d(channels, channels, kernel_size=1)
        self.key = nn.Conv2d(channels, channels, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)

        # Frequency-domain weights (trainable)
        self.freq_weights = nn.Parameter(torch.ones(channels, dtype=torch.float32))

        # Output projection
        self.output = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        # Input: x (batch_size, channels, height, width)
        batch_size, channels, height, width = x.size()

        # Frequency attention: Apply FFT to convert features to frequency domain
        freq_domain = torch.fft.fft2(x, dim=(-2, -1))
        freq_magnitude = torch.abs(freq_domain)  # Magnitude of frequency components

        # Modulate frequency components with learnable weights
        freq_attention = freq_magnitude * self.freq_weights.view(1, -1, 1, 1)

        # Return to spatial domain
        refined_freq = freq_attention * torch.exp(1j * torch.angle(freq_domain))
        refined_features = torch.fft.ifft2(refined_freq, dim=(-2, -1)).real  # Take the real part

        # Self-attention mechanism
        q = self.query(refined_features)  # Query
        k = self.key(refined_features)  # Key
        v = self.value(refined_features)  # Value

        # Compute attention map (scaled dot product)
        attention_map = torch.softmax(torch.matmul(q.view(batch_size, channels, -1).transpose(1, 2),
                                                   k.view(batch_size, channels, -1)) / (channels ** 0.5), dim=-1)

        # Apply attention weights to the value features
        v_flat = v.view(batch_size, channels, -1)
        refined_attention = torch.matmul(attention_map, v_flat.transpose(1, 2)).transpose(1, 2)

        # Reshape back to spatial dimensions
        refined_attention = refined_attention.view(batch_size, channels, height, width)

        # Apply the output projection
        output = self.output(refined_attention)

        return output





class FrequencySelfAttention4(nn.Module):   #SAM+CAM+FAM3+Selfattn
    def __init__(self, channels):
        super(FrequencySelfAttention4, self).__init__()
        # Key, Query, and Value projections
        self.query = nn.Conv2d(channels, channels, kernel_size=1)
        self.key = nn.Conv2d(channels, channels, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)

        # Frequency-domain weights (trainable)
        self.freq_weights = nn.Parameter(torch.ones(channels, dtype=torch.float32))

        # Output projection
        self.output = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x, guiding_map0):

        guiding_map0 = F.interpolate(guiding_map0, x.size()[2:], mode='bilinear', align_corners=True)
        guiding_map = F.sigmoid(guiding_map0)

        # Input: x (batch_size, channels, height, width)
        batch_size, channels, height, width = x.size()

        # Frequency attention: Apply FFT to convert features to frequency domain
        freq_domain = torch.fft.fft2(x, dim=(-2, -1))
        freq_magnitude = torch.abs(freq_domain)  # Magnitude of frequency components

        # Modulate frequency components with learnable weights
        freq_attention = freq_magnitude * self.freq_weights.view(1, -1, 1, 1)

        # Return to spatial domain
        refined_freq = freq_attention * torch.exp(1j * torch.angle(freq_domain))
        refined_features = torch.fft.ifft2(refined_freq, dim=(-2, -1)).real  # Take the real part

        # Self-attention mechanism
        q = self.query(refined_features)  # Query
        k = self.key(refined_features)  # Key
        v = self.value(refined_features)  # Value

        q = q * (1+ guiding_map)
        k = k * (1+ guiding_map)
        v = v * (1+ guiding_map)

        # Compute attention map (scaled dot product)
        attention_map = torch.softmax(torch.matmul(q.view(batch_size, channels, -1).transpose(1, 2),
                                                   k.view(batch_size, channels, -1)) / (channels ** 0.5), dim=-1)

        # Apply attention weights to the value features
        v_flat = v.view(batch_size, channels, -1)
        refined_attention = torch.matmul(attention_map, v_flat.transpose(1, 2)).transpose(1, 2)

        # Reshape back to spatial dimensions
        refined_attention = refined_attention.view(batch_size, channels, height, width)

        # Apply the output projection
        output = self.output(refined_attention)

        return output





class ChangeGuideModule(nn.Module):
    def __init__(self, in_dim, device=None):
        super(ChangeGuideModule, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1, device=device)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1, device=device)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1, device=device)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, guiding_map0):
        m_batchsize, C, height, width = x.size()
        # print ("guiding_map0 first ", guiding_map0.shape)

        guiding_map0 = F.interpolate(guiding_map0, x.size()[2:], mode='bilinear', align_corners=True)
        # print ("guiding_map0", guiding_map0.shape)

        guiding_map = F.sigmoid(guiding_map0)
        # print ("guiding_map", guiding_map.shape)
        # print ("x shape: ", x.shape)

        #query = self.query_conv(x) * (1 + guiding_map)

        query = self.query_conv(x) 
        # print ("query shape: ", query.shape)
        query = query * (1 + guiding_map)
        proj_query = query.view(m_batchsize, -1, width*height).permute(0, 2, 1)
        
        key = self.key_conv(x) * (1 + guiding_map)
        proj_key = key.view(m_batchsize, -1, width*height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        self.energy = energy
        self.attention = attention

        value = self.value_conv(x) * (1 + guiding_map)
        proj_value = value.view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        # print ("out size: ", out.shape)

        out = self.gamma * out + x
        # print ("out2 size: ", out.shape)
        # print (self.gamma)
        # print ("out min: ", out.min(), "out max: ", out.max())
        # print ("out mean:", out.mean(), "out std: ", out.std())

        return out

################## My Idea ######################

# class My_GuideModule2(nn.Module):
#     def __init__(self, in_channels, device=None):
#         super(My_GuideModule2,self).__init__()

#         self.in_channels = in_channels
#         self.ca = ChannelAttention(self.in_channels)
#         self.my_sa = My_SpatialAttention ()
#         self.my_fa = FrequencyAttention(self.in_channels)

#     def forward(self, x, guiding_map0):
#         skip = x
#         x = x*self.ca(x)
#         x = x*self.my_sa(x, guiding_map0)
#         x = self.my_fa(x)
#         x = x+skip
#         return x
    


class My_GuideModule2(nn.Module):     # I removed the FreqAttention
    def __init__(self, in_channels, device=None):
        super(My_GuideModule2,self).__init__()

        self.in_channels = in_channels
        self.ca = ChannelAttention(self.in_channels)
        self.my_sa = My_SpatialAttention ()
        self.my_fa = FrequencyAttention2(self.in_channels)
        # self.my_cg = ChangeGuideModule(self.in_channels)

    def forward(self, x, guiding_map0):
        skip = x
        x = x*self.ca(x)
        x = x*self.my_sa(x, guiding_map0)
        x = self.my_fa(x)
        # x = self.my_cg(x, guiding_map0)
        x = x+skip
        return x
    




# class My_GuideModule3(nn.Module):   # The three modules are put into Parallel 
#     def __init__(self, in_channels, device=None):
#         super(My_GuideModule3,self).__init__()

#         self.in_channels = in_channels
#         self.ca = ChannelAttention(self.in_channels)
#         self.my_sa = My_SpatialAttention ()
#         self.my_fa = FrequencyAttention(self.in_channels)

#     def forward(self, x, guiding_map0):
#         skip = x
#         x1 = x*self.ca(x)
#         x2 = x*self.my_sa(x, guiding_map0)
#         x3 = self.my_fa(x)
#         x = x1+x2+x3
#         x = x+skip
#         return x
    






if __name__ == "__main__":
    # x = torch.rand(1,3,256, 256)
    # gmap = torch.rand(1,1,16,16)
    # model = My_SpatialAttention()
    # y = model(x, gmap)
    # # x = x*model(x)
    # print (y.shape)


    x = torch.rand(1,8,256, 256)
    gmap = torch.rand(1,1,16,16)
    model = FrequencyAttention2(8)
    y = model(x)
    # x = x*model(x)
    print (y.shape)
    print (y.min(), y.max())


    # x = torch.rand(1,8,256, 256)
    # # gmap = torch.rand(1,1,16,16)
    # model = FrequencyAttention(8)
    # y = model(x)
    # # x = x*model(x)
    # print (y.shape)