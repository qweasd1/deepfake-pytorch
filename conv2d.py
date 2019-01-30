from torch import nn

kernel_size = 5
stride = 2


def conv2d(in_channels, out_channels):
    half_kernel_size = kernel_size // 2
    right_pad = max(0,half_kernel_size - 1)
    return nn.Sequential(
        nn.ZeroPad2d((half_kernel_size, right_pad, kernel_size // 2, right_pad)),
        nn.Conv2d(in_channels, out_channels, kernel_size, stride),
        nn.LeakyReLU(0.1, inplace=True)
    )
