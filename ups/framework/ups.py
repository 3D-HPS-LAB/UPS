import torch
import torch.nn as nn
import soft_renderer as sr
from loguru import logger

from ..utils import constants

from ..utils.smpl import SMPL
from ..utils.geometry import convert_camT_to_proj_mat
from ..utils.renderer import Renderer


class UPS(nn.Module):
    def __init__(
            self,
            basic_s_weight=0.,
            basic_c_weight=0.,
            ups_s_weight=0.1,
            ups_c_weight=0.1,
            ups_train_only=False,
    ):
        super(UPS, self).__init__()

        self.gamma_val = 1.0e-1
        self.sigma_val = 1.0e-7
        self.background_color = [0.0, 0.0, 0.0]
        self.light_intensity_ambient = 1.0
        self.light_intensity_directionals = 0
        self.img_size = constants.IMG_RES
        self.smpl = SMPL(constants.SMPL_MODEL_DIR, batch_size=1, create_transl=False)
        self.smpl_faces = torch.from_numpy(self.smpl.faces.astype('int32'))[None].cuda()
        self.basic_s_loss = nn.L1Loss()

        self.renderer = Renderer(
            focal_length=5000,
            img_res=224,
            faces=self.smpl.faces,
            mesh_color='blue',
        )

        self.basic_s_weight = basic_s_weight
        self.basic_c_weight = basic_c_weight
        self.ups_s_weight = ups_s_weight
        self.ups_c_weight = ups_c_weight
        self.ups_train_only = ups_train_only

    def ups_losses(self, imgnames, vertices, cam_t,
                   skin_sigma, grph_skin_gt,
                   grph_clothing_gt, grph_clothing_dist_gt, clothing_sigma,
                   smpl_skin_texture_gt, smpl_clothing_texture_gt,
                   grph_skin_valid_labels, grph_clothing_valid_labels, start_ups):

        batch_size = vertices.shape[0]
        # i.e. 'skin' + 'clothing', 2 labels
        rend_dim = len([label for label in constants.UPS_LABEL if label != 'background'])

        # (batch_size,) skin loss
        loss_basic_s = torch.zeros(batch_size, device=vertices.device)
        # (batch_size,) clothing loss
        loss_basic_c = torch.zeros(batch_size, device=vertices.device)
        # (batch_size,) sigma loss
        loss_ups_s = torch.zeros(batch_size, device=vertices.device)
        # (batch_size,) sigma loss
        loss_ups_c = torch.zeros(batch_size, device=vertices.device)

        # Late start of UPS Losses
        if not start_ups:
            # logger.warning(f'ups loss has not started')
            return loss_basic_s.mean(), loss_basic_c.mean(), loss_ups_s.mean(), loss_ups_c.mean()

        # torch.Size([batch_size, 224, 224])
        grph_skin_gt = grph_skin_gt.mean(3).float()

        # torch.Size([batch_size, 3, 224, 224])
        grph_clothing_gt = grph_clothing_gt.permute(0, 3, 1, 2)

        # torch.Size([batch_size, 3, 224, 224])
        grph_clothing_dist_gt = grph_clothing_dist_gt.permute(0, 3, 1, 2)

        # torch.Size([batch_size, rend_dim, 6890, 3])
        textures = torch.cat((smpl_skin_texture_gt.unsqueeze(1), smpl_clothing_texture_gt.unsqueeze(1)), dim=1)

        # torch.Size([batch_size, 3, 4])
        P = convert_camT_to_proj_mat(cam_t.cpu()).cuda()

        # self.smpl_faces: torch.Size([1, 13776, 3])
        # batch_smpl_faces: torch.Size([rend_dim * batch_size, 13776, 3])
        batch_smpl_faces = self.smpl_faces.expand(rend_dim * batch_size, self.smpl_faces.shape[1],
                                                  self.smpl_faces.shape[2])
        # vertices: torch.Size([batch_size, 6890, 3])
        # batch_vertices: torch.Size([rend_dim * batch_size, 6890, 3])
        batch_vertices = torch.repeat_interleave(vertices, repeats=rend_dim,
                                                 dim=0)
        # torch.Size([rend_dim * batch_size, 3, 4])
        batch_P = torch.repeat_interleave(P, repeats=rend_dim,
                                          dim=0)
        # textures: torch.Size([batch_size, rend_dim, 6890, 3])
        # batch_textures: torch.Size([rend_dim * batch_size, 6890, 3])
        batch_textures = textures.view(rend_dim * batch_size, textures.shape[2], textures.shape[3])

        # render the labels
        renderer = sr.SoftRenderer(P=batch_P,
                                   camera_mode='projection',
                                   gamma_val=self.gamma_val,
                                   sigma_val=self.sigma_val,
                                   orig_size=self.img_size,
                                   image_size=self.img_size,
                                   background_color=self.background_color,
                                   light_intensity_ambient=self.light_intensity_ambient,
                                   light_intensity_directionals=self.light_intensity_directionals,
                                   texture_type='vertex')

        # batch_vertices is torch.Size([rend_dim * batch_size, 6890, 3])
        # batch_smpl_faces is torch.Size([rend_dim * batch_size, 13776, 3])
        # batch_textures is torch.Size([rend_dim * batch_size, 6890, 3]),
        #       for each image, contains 2 labels(skin + clothing),each label contains 6890 vertexs, has 3 RGB color
        # rend_out is (288, 4, 224, 224), for each image each label, contains 4 rendered channels, 1 gray channel, 3 RGB channel
        #       rend_out[i] ~ rend_out[i + 8] contains 9 labels(1 mc + 8 c)
        #       for each rend_out[i], contains (3, 224, 224) RGB & (1, 224, 224) gray
        rend_out = renderer(batch_vertices, batch_smpl_faces, batch_textures, texture_type='vertex')

        # Calculate loss for individual sample in batch to avoid computing loss
        # for samples without graphonomy labels
        for idx in range(batch_size):

            if len(grph_skin_valid_labels[idx]) == 0 and len(grph_clothing_valid_labels[idx]) == 0:
                # logger.warning(f'No valid labels')
                continue
            # torch.Size([1, 224, 224])
            cur_grph_skin_gt = grph_skin_gt[None, idx]
            # torch.Size([3, 224, 224])
            cur_clothing_gt = grph_clothing_gt[idx]
            # torch.Size([1, 3, 224, 224])
            cur_clothing_dist_gt = grph_clothing_dist_gt[None, idx]
            # torch.Size([1, 224, 224])
            cur_skin_sigma = skin_sigma[idx]
            cur_clothing_sigma = clothing_sigma[idx]

            start_index = rend_dim * idx
            # cur_skin_rend_out: torch.Size([4, 224, 224])
            # cur_clothing_rend_out: torch.Size([4, 224, 224])
            cur_skin_rend_out, cur_clothing_rend_out = rend_out[start_index:start_index + rend_dim]

            cur_skin_rend_out = cur_skin_rend_out[:3].mean(0).unsqueeze(0)

            # Don't compute UPS loss if all loss_weights are zero
            if self.basic_s_weight != 0 and self.basic_c_weight != 0:
                # cur_skin_rend_out: torch.Size([1, 224, 224])
                # cur_grph_skin_gt: torch.Size([1, 224, 224])
                loss_basic_s[idx] = self.basic_s_loss(cur_skin_rend_out, cur_grph_skin_gt)
                # cur_clothing_rend_out: torch.Size([4, 224, 224])
                # cur_clothing_gt: torch.Size([3, 224, 224])
                # cur_clothing_dist_gt: torch.Size([1, 3, 224, 224])
                loss_basic_c[idx] = basic_c_loss(cur_clothing_rend_out, cur_clothing_gt, cur_clothing_dist_gt)

            if self.ups_s_weight != 0 and self.ups_c_weight != 0:
                loss_ups_s[idx] = ups_s_loss(cur_skin_rend_out, cur_grph_skin_gt, cur_skin_sigma)
                loss_ups_c[idx] = ups_c_loss(cur_clothing_rend_out, cur_clothing_gt,
                                             cur_clothing_dist_gt, cur_clothing_sigma)

            if torch.isnan(loss_basic_s[idx]) or torch.isnan(loss_basic_s[idx]) or \
                    torch.isinf(loss_basic_c[idx]) or torch.isinf(loss_basic_c[idx]) or \
                    torch.isinf(loss_ups_s[idx]) or torch.isinf(loss_ups_s[idx]) or \
                    torch.isinf(loss_ups_c[idx]) or torch.isinf(loss_ups_c[idx]):
                logger.warning(f'loss is nan for {imgnames[idx]}')
                loss_basic_s[idx] = 0.
                loss_basic_c[idx] = 0.
                loss_ups_s[idx] = 0.
                loss_ups_c[idx] = 0.

        return loss_basic_s.mean(), loss_basic_c.mean(), loss_ups_s.mean(), loss_ups_c.mean()

    def forward(self, pred, gt, start_ups):

        # Get predictions
        pred_cam_t = pred['pred_cam_t']
        pred_vertices = pred['smpl_vertices']
        pred_skin_sigma = pred['pred_skin_sigma']
        pred_clothing_sigma = pred['pred_clothing_sigma']

        # (224, 224, 3)
        grph_skin_gt = gt['grph_skin_gt']
        # (224, 224, 3)
        grph_clothing_gt = gt['grph_clothing_gt']
        # (224, 224, 3)
        grph_clothing_dist_gt = gt['grph_clothing_dist_gt']
        # (24,)
        grph_skin_valid_labels = gt['grph_skin_valid_labels']
        # (24,)
        grph_clothing_valid_labels = gt['grph_clothing_valid_labels']

        # (6890, 3) smpl skin vertex texture
        smpl_skin_texture_gt = gt['smpl_skin_texture_gt']
        # (6890, 3) smpl clothing vertex texture
        smpl_clothing_texture_gt = gt['smpl_clothing_texture_gt']

        smpl_segm_valid_idx = []

        for idx, has_smpl_skin_parts in enumerate(gt['has_smpl_skin_parts']):
            if has_smpl_skin_parts:
                smpl_segm_valid_idx.append(idx)

        if len(smpl_segm_valid_idx) != 0:
            loss_basic_s, loss_basic_c, loss_ups_s, loss_ups_c = \
                self.ups_losses([gt['imgname'][i] for i in smpl_segm_valid_idx],
                                pred_vertices[smpl_segm_valid_idx],
                                pred_cam_t[smpl_segm_valid_idx],
                                pred_skin_sigma[smpl_segm_valid_idx],
                                grph_skin_gt[smpl_segm_valid_idx],
                                grph_clothing_gt[smpl_segm_valid_idx],
                                grph_clothing_dist_gt[smpl_segm_valid_idx],
                                pred_clothing_sigma[smpl_segm_valid_idx],
                                smpl_skin_texture_gt[smpl_segm_valid_idx],
                                smpl_clothing_texture_gt[smpl_segm_valid_idx],
                                grph_skin_valid_labels[smpl_segm_valid_idx],
                                grph_clothing_valid_labels[smpl_segm_valid_idx],
                                start_ups,
                                )
        else:
            loss_basic_s = torch.FloatTensor(1).fill_(0.).to(pred_vertices.device)
            loss_basic_c = torch.FloatTensor(1).fill_(0.).to(pred_vertices.device)
            loss_ups_s = torch.FloatTensor(1).fill_(0.).to(pred_vertices.device)
            loss_ups_c = torch.FloatTensor(1).fill_(0.).to(pred_vertices.device)

        loss_basic_s *= self.basic_s_weight
        loss_basic_c *= self.basic_c_weight
        loss_ups_s *= self.ups_s_weight
        loss_ups_c *= self.ups_c_weight

        loss_dict = {
            'loss/loss_basic_s': loss_basic_s,
            'loss/loss_basic_c': loss_basic_c,
            'loss/loss_ups_s': loss_ups_s,
            'loss/loss_ups_c': loss_ups_c,
        }

        loss = sum(loss for loss in loss_dict.values())

        loss *= self.loss_weight

        loss_dict['loss/total_loss'] = loss

        return loss, loss_dict


def basic_c_loss(pred, target, dist):
    """
    :param pred: torch.Size([4, 224, 224])
    :param target: torch.Size([3, 224, 224])
    :param dist: torch.Size([1, 3, 224, 224])
    :return: basic_c loss
    """
    dist = dist[0][0] / torch.max(dist)
    pred = pred[:3].mean(0)
    loss = pred * dist
    return torch.mean(loss)


def ups_s_loss(pred, target, sigma):
    """
    1 / sigma * |target - pred| + ln(sigma)
    :param pred: torch.Size([1, 224, 224])
    :param target: torch.Size([1, 224, 224])
    :param sigma: torch.Size([1, 224, 224]), represents log(sigma^2)
    :return: ups_s loss
    """
    sigma = torch.exp(sigma / 2) + 1
    loss = 1 / sigma * torch.abs(target - pred) + torch.log(sigma)
    return torch.mean(loss)


def ups_c_loss(pred, target, dist, sigma):
    """
    1 / sigma * pred * dist + ln(sigma)
    :param pred: torch.Size([4, 224, 224])
    :param target: torch.Size([3, 224, 224])
    :param dist: torch.Size([1, 3, 224, 224])
    :param sigma: torch.Size([1, 224, 224]), represents log(sigma^2)
    :return: ups_c loss
    """
    sigma = torch.exp(sigma / 2) + 1
    dist = dist[0][0] / torch.max(dist)
    pred = pred[:3].mean(0)
    loss = 1 / sigma * pred * dist + torch.log(sigma)
    return torch.mean(loss)
