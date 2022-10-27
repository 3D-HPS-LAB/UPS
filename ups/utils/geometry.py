import torch

def convert_camT_to_proj_mat(cam_t, focal_length=5000., img_size=224):
    R = torch.eye(3)
    RT = torch.zeros((cam_t.shape[0], 3, 4))  # camera extrinsic parameter
    RT[:, :, 0:3] = R
    RT[:, :, 3] = cam_t
    RT[:, 1, 1] *= -1.0  # Rotate around Y axis
    RT[:, 1, 3] *= -1.0  # Negate the y translation
    K = torch.Tensor([[focal_length, 0, img_size / 2],
                      [0, focal_length, img_size / 2],
                      [0, 0, 1]])  # camera intrinsic parameter
    P = torch.matmul(K, RT)
    return P
