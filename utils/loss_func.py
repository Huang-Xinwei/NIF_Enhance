import torch

def darkchannel_loss(input, gt, window=14):
    """Get the dark channel prior in a batch of (RGB) images.

    Parameters`
    -----------
    I:  an B * C * M * N tensor containing data ([0, L-1]) in the image where
        B is the batchsize, C is the number of channel, M is the height, N is the width.
    w:  window size

    Return
    -----------
    An B * M * N tensor for the dark channel prior ([0, L-1]).
    """
    minpool = torch.nn.MaxPool3d(kernel_size=(3, window, window), stride=1) # 3 represent R/G/B chaneels
    input = torch.nn.functional.pad(input, (window//2, window//2-1, window//2, window//2-1), 'replicate')
    gt = torch.nn.functional.pad(gt, (window//2, window//2-1, window//2, window//2-1), 'replicate')
    output_pred = -minpool(-input)
    output_gt = -minpool(-gt)
    loss = torch.sum(torch.pow(output_pred - output_gt, 2))
    return loss


def RGB2XYZ(input):
    X = (0.4124 * input[:,0,:,:] + 0.3576 * input[:,1,:,:] + 0.1805 * input[:,2,:,:])/95.0489
    Y = (0.2126 * input[:,0,:,:] + 0.7152 * input[:,1,:,:] + 0.0722 * input[:,2,:,:])/100
    Z = (0.0193 * input[:,0,:,:] + 0.1192 * input[:,1,:,:] + 0.9509 * input[:,2,:,:])/108.8840
    return X, Y, Z


def f_lab(input):
    img = 0.008856*torch.ones(input.shape)
    img = img.to(device=torch.device('cuda'), dtype=torch.float32)
    pare = torch.gt(input, img)
    if torch.equal(pare, torch.ones_like(input, dtype=bool)):
        return torch.pow(input, 1/3)
    else:
        return 7.787*input + 0.1379


def cie76_part_loss(input, target):
    """
    # L = 116*f_lab(y) - 16
    # a = 500*(f_lab(x) - f_lab(y))
    # b = 200*(f_lab(y) - f_lab(z))
    """
    x, y, z = RGB2XYZ(input)
    x = f_lab(x)
    y = f_lab(y)
    z = f_lab(z)
    x_hat, y_hat, z_hat = RGB2XYZ(target)
    x_hat = f_lab(x_hat)
    y_hat = f_lab(y_hat)
    z_hat = f_lab(z_hat)
    delta_A_2 = 500*500*torch.pow(x - y - x_hat + y_hat, 2)
    delta_B_2 = 200*200*torch.pow(y - z - y_hat + z_hat, 2)
    delta_L_2 = 116*116*torch.pow(y - y_hat, 2)
    delta_sum = torch.pow(delta_A_2 + delta_B_2 + delta_L_2, 1/2)
    return torch.sum(delta_sum)

