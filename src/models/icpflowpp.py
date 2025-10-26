import math
import numpy as np

import dztimer
import torch
import torch.nn as nn
import pytorch3d.ops as pytorch3d_ops
import torch.nn.functional as F
import cuml
import hdbscan
import matplotlib.pyplot as plt
from assets.cuda.histgram import histgram

from .basic import cal_pose0to1

import warnings
warnings.filterwarnings('ignore')
np.set_printoptions(suppress=True)

class ICPFlowpp(nn.Module):
    def __init__(self, 
                 point_cloud_range = [-51.2, -51.2, -3, 51.2, 51.2, 3], 
                #  point_cloud_range = [-50, -50, -3, 50, 50, 3], 
                 flow_range=[-2.0, -2.0, -0.2, 2.0, 2.0, 0.2],
                 voxel_size=[0.1, 0.1, 0.1],
                 topk=3,
                 **kwargs):
        super().__init__()
         
        self.point_cloud_range = point_cloud_range
        self.flow_range = flow_range
        self.voxel_size = voxel_size
        self.topk = topk
         
        self.timer = dztimer.Timing()
        self.timer.start("Total")
    
        eps = 1e-8
        self.bins_x = nn.Parameter(torch.arange(flow_range[0], flow_range[3] + eps, voxel_size[0]), requires_grad=False)
        self.bins_y = nn.Parameter(torch.arange(flow_range[1], flow_range[4] + eps, voxel_size[1]), requires_grad=False)
        self.bins_z = nn.Parameter(torch.arange(flow_range[2], flow_range[5] + eps, voxel_size[2]), requires_grad=False)
        self.d, self.h, self.w = len(self.bins_z), len(self.bins_y), len(self.bins_x)
        # https://docs.rapids.ai/api/cuml/stable/api/#cuml.cluster.hdbscan.HDBSCAN
        # self.clusterer = cuml.cluster.hdbscan.HDBSCAN(min_cluster_size=50, output_type='cupy')
        self.clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1., approx_min_span_tree=True,
                                    gen_min_span_tree=True, leaf_size=100,
                                    metric='euclidean', min_cluster_size=20, min_samples=None)

    def visualize(self, pc0_np, pc1_np, label0_np, label1_np, flow_np, flow_gt_np, png_name=None):
        pc01_np = pc0_np + flow_np
        pc01_gt_np = pc0_np + flow_gt_np
        fig, ax = plt.subplots(figsize=(24, 24))
        ax.scatter(pc0_np[:, 0], pc0_np[:, 1], c='g', s=1, label='pc0')
        ax.scatter(pc1_np[:, 0], pc1_np[:, 1], c='b', s=1, label='pc1')
        ax.set_xlim(-51.2, 51.2)
        ax.set_ylim(-51.2, 51.2)
        if png_name is None:
            plt.show()
        else:
            plt.savefig(f'{png_name}_pcds.png')
        plt.close()

        fig, ax = plt.subplots(figsize=(24, 24))
        ax.scatter(pc0_np[:, 0], pc0_np[:, 1], c=label0_np, s=1, label='pc1')
        # ax.scatter(pc01_np[:, 0], pc01_np[:, 1], c='r', s=1, label='pc01')
        ax.set_xlim(-51.2, 51.2)
        ax.set_ylim(-51.2, 51.2)
        if png_name is None:
            plt.show()
        else:
            plt.savefig(f'{png_name}_pcd0_label.png')
        plt.close()

        # fig, ax = plt.subplots(figsize=(24, 24))
        # # for i in range(len(flow_np)):
        # for i in np.random.randint(0, len(flow_np), size=(10000,)):
        #     ax.arrow(pc0_np[i, 0], pc0_np[i, 1], flow_np[i, 0], flow_np[i, 1], head_width=0.2, head_length=0.2, fc='red', ec='b')
        #     ax.scatter(pc0_np[i, 0], pc0_np[i, 1], c='g', s=0.2, label='pc01')
        # # ax.scatter(pc01_np[:, 0], pc01_np[:, 1], c='b', s=1, label='pc1')
        # ax.set_xlim(-51.2, 51.2)
        # ax.set_ylim(-51.2, 51.2)
        # plt.savefig(f'{png_name}_flow.png')
        # # plt.show()
        # plt.close()

        fig, ax = plt.subplots(figsize=(24, 24))
        ax.scatter(pc1_np[:, 0], pc1_np[:, 1], c='b', s=1, label='pc1')
        ax.scatter(pc01_np[:, 0], pc01_np[:, 1], c='r', s=1, label='prediction')
        # ax.scatter(pc01_gt_np[:, 0], pc01_gt_np[:, 1], c='b', s=1, label='gt')
        ax.set_xlim(-51.2, 51.2)
        ax.set_ylim(-51.2, 51.2)
        if png_name is None:
            plt.show()
        else:
            plt.savefig(f'{png_name}_pd.png')
        plt.close()

        error = np.linalg.norm(flow_gt_np - flow_np, axis=1)
        error = np.linalg.norm(flow_gt_np - flow_np, axis=1)
        mask = error >0.5
        fig, ax = plt.subplots(figsize=(24, 24))
        ax.scatter(pc0_np[:, 0], pc0_np[:, 1], c='g', s=1, label='pc0')
        ax.scatter(pc01_np[:, 0], pc01_np[:, 1], c='b', s=1, label='pd')
        ax.scatter(pc01_np[mask, 0], pc01_np[mask, 1], c='r', s=1, label='error')
        ax.set_xlim(-51.2, 51.2)
        ax.set_ylim(-51.2, 51.2)
        if png_name is None:
            plt.show()
        else:
            plt.savefig(f'{png_name}_error.png')
        plt.close()

    def crop_pcds(self, pc):
        """
        Limit the point cloud to the given range.
        """
        mask = (pc[:, 0] >= self.point_cloud_range[0]) & (pc[:, 0] <= self.point_cloud_range[3]) & \
               (pc[:, 1] >= self.point_cloud_range[1]) & (pc[:, 1] <= self.point_cloud_range[4]) & \
               (pc[:, 2] >= self.point_cloud_range[2]) & (pc[:, 2] <= self.point_cloud_range[5])
        # mask = (pc[:, 0] >= self.point_cloud_range[0]) & (pc[:, 0] <= self.point_cloud_range[3]) & \
        #        (pc[:, 1] >= self.point_cloud_range[1]) & (pc[:, 1] <= self.point_cloud_range[4]) 
        return pc[mask], mask
 
    def downsample(self, pc):
        # crop the pcds and then downsample
        voxels = pc.clone()
        voxels[:, 0] -= self.point_cloud_range[0]
        voxels[:, 1] -= self.point_cloud_range[1]
        voxels[:, 2] -= self.point_cloud_range[2]

        voxels[:, 0] /= self.voxel_size[0]
        voxels[:, 1] /= self.voxel_size[1]
        voxels[:, 2] /= self.voxel_size[2]
        voxels = voxels.to(torch.int64)
        voxel_coords, idxs, counts = voxels.unique(return_inverse=True, return_counts=True, dim=0)
        l = len(voxel_coords)
        voxels_mean = torch.zeros((l, 3), device=voxels.device)
        voxels_mean[:, 0].scatter_add_(dim=0, index=idxs, src=pc[:, 0])
        voxels_mean[:, 1].scatter_add_(dim=0, index=idxs, src=pc[:, 1])
        voxels_mean[:, 2].scatter_add_(dim=0, index=idxs, src=pc[:, 2])
        voxels_mean /= counts[:, None]
        # print('downsample: ', pc.shape, voxels_mean.shape)
        return voxels_mean

    def nms(self, x, kernel_size=3):
        x = x[:, None, :, :, :].float()
        xp = torch.nn.functional.max_pool3d(x, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
        mask = (x == xp).float().clamp(min=0.0)
        xp = x * mask
        return xp[:, 0, :, :, :]

    def select_topk_offset_per_cluster(self, histgram_t):
        histgram_t = self.nms(histgram_t)
        topk_max, topk_argmax = histgram_t.view(len(histgram_t), self.d*self.h*self.w).topk(k=self.topk, dim=1) # [l, k]
        topk_x, topk_y, topk_z = topk_argmax%self.w, topk_argmax//self.w%self.h, topk_argmax//self.h//self.w%self.d
        return topk_max, topk_x, topk_y, topk_z 

    def flow_calculation(self, pc0, pc1, offsets, unqs, idxs_inverse):    
        # sanity check
        assert len(offsets) == len(idxs_inverse)
        assert idxs_inverse.min()==0
        assert idxs_inverse.max()+1== len(unqs)

        # calculate nn for each point, compensated by the topk offsets
        pc01 = pc0[:, None, :] + offsets # [m, topk, 3]
        dis_topk, idxs_topk, _ = pytorch3d_ops.knn_points(pc01.view(1, len(pc0)*self.topk, 3), pc1[None], lengths1=None, lengths2=None, K=1, return_nn=False, return_sorted=False) # [1, m, k]
        dis_topk, idxs_topk = dis_topk[0].view(len(pc0), self.topk), idxs_topk[0].view(len(pc0), self.topk) # [m, k]

        # accumulate errors per cluster at the topk offset positions and pick up the offset that leads to the minimal error per cluster
        idxs_flatten = idxs_inverse.repeat_interleave(self.topk) * self.topk + torch.arange(0, self.topk, device=offsets.device).repeat(len(pc0)) # row idx: wich cluster a point belongs to; col idx: topk
        clusters = torch.zeros(size=(len(unqs) * self.topk, ), device=offsets.device) # [l, k]
        clusters.scatter_add_(dim=0, index=idxs_flatten, src=dis_topk.flatten())

        clusters_min, clusters_argmin = clusters.view(len(unqs), self.topk).min(dim=-1) # [l]
        # print('cluster min: ', clusters_min.shape, clusters_min.max(), clusters_min.min())
        # print('cluster argmin: ', clusters_argmin.shape, clusters_argmin.max(), clusters_argmin.min())
        clusters_argmin_pts = clusters_argmin[idxs_inverse] # [m]

        offsets_x = torch.gather(offsets[:, :, 0], index=clusters_argmin_pts[:, None], dim=1) # [m, 1]
        offsets_y = torch.gather(offsets[:, :, 1], index=clusters_argmin_pts[:, None], dim=1) # [m, 1]
        offsets_z = torch.gather(offsets[:, :, 2], index=clusters_argmin_pts[:, None], dim=1) # [m, 1]
        offsets_min = torch.cat([offsets_x, offsets_y, offsets_z], dim=1) 
        return offsets_min

    def _model_forward(self, points_src, points_dst, labels_src):
        self.timer[1][0].start("histogram calculation")
        with torch.no_grad():

            unqs, idxs_inverse, counts = torch.unique(labels_src, return_inverse=True, return_counts=True)
            idxs_inverse = idxs_inverse.to(torch.int32)

            points_dst_ = self.downsample(points_dst)

            histgram_t = torch.zeros([len(unqs), self.d, self.h, self.w], dtype=torch.int32).to(points_src.device).contiguous()    
            histgram.histgram_func(points_src, points_dst_, idxs_inverse, \
                                    histgram_t, \
                                     self.flow_range[0], self.flow_range[1], self.flow_range[2], \
                                         self.flow_range[3], self.flow_range[4], self.flow_range[5], \
                                             self.w, self.h, self.d)    
        self.timer[1][0].stop()

        self.timer[1][1].start("topk selection")
        topk_max, topk_x, topk_y, topk_z = self.select_topk_offset_per_cluster(histgram_t) # [l, k]
        topk_ratio = topk_max/counts[:, None]

        # let op: corner case: 1. a cluster may receive few votes or proportionally a low ratio; 2. non-clustered ponts (label==0 in this codebase).
        invalid = torch.logical_or(topk_max<10,  topk_ratio<0.1)
        invalid[0, :] = True
        topk_x[invalid] = self.w//2 # set zero offset
        topk_y[invalid] = self.h//2 # set zero offset
        topk_z[invalid] = self.d//2 # set zero offset

        offsets = torch.stack([ self.bins_x[topk_x], self.bins_y[topk_y], self.bins_z[topk_z] ], dim=2)
        offsets = offsets[idxs_inverse] # [m, k, 3]
        self.timer[1][1].stop()
        

        self.timer[1][2].start("flow calculation")
        flows = self.flow_calculation(points_src, points_dst, offsets, unqs, idxs_inverse) # [l, k]
        self.timer[1][2].stop()
 
        return flows
    
    
    def forward(self, batch):
        """
        input: using the batch from dataloader, which is a dict
               Detail: [pc0, pc1, pose0, pose1]
        output: the predicted flow, pose_flow, and the valid point index of pc0
        """
        # print(f'processing sample - scene id {batch["scene_id"]}, timestamp {batch["timestamp"]} ')
        self.timer[0].start("Data Preprocess")
        batch_sizes = len(batch["pose0"])
        assert batch_sizes==1

        for batch_id in range(batch_sizes):
            pc0 = batch["pc0"][batch_id]
            pc1 = batch["pc1"][batch_id]
            pc0_selected, mask0 = self.crop_pcds(pc0)
            pc1_selected, mask1 = self.crop_pcds(pc1)

            self.timer[0][0].start("pose")
            with torch.no_grad():
                if 'ego_motion' in batch:
                    pose_0to1 = batch['ego_motion'][batch_id]
                else:
                    pose_0to1 = cal_pose0to1(batch["pose0"][batch_id], batch["pose1"][batch_id])

            self.timer[0][0].stop()
            
            self.timer[0][1].start("transform")
            # transform selected_pc0 to pc1
            pc0_transformed = pc0_selected @ pose_0to1[:3, :3].T + pose_0to1[:3, 3]
            self.timer[0][1].stop()

            self.timer[0][1].start("transform")

            self.timer[0][2].start("clustering")
            pc0_transformed_np = pc0_transformed.clone().cpu().numpy()
            self.clusterer.fit(pc0_transformed_np)
            label0 = self.clusterer.labels_ + 1
            label0 = torch.as_tensor(label0, device=pc0_transformed.device)
            self.timer[0][2].stop()

            self.timer[0].stop()
        
            self.timer[1].start("Model Forward")
            flow_selected = self._model_forward(pc0_transformed, pc1_selected, label0)
            self.timer[1].stop()
        
            # self.timer.print(random_colors=True, bold=True)
            # # # # # visualize results (batch_size==1 during val/test)
            # pc0_np = pc0_transformed.clone().cpu().numpy()
            # pc1_np = pc1_selected.clone().cpu().numpy()
            # label0_np = label0.clone().cpu().numpy()
            # label1_np = None
            # flow_np = flow_selected.clone().cpu().numpy()

            # flow_pose = (pc0_transformed - pc0_selected)
            # flow_pose_np = flow_pose.clone().cpu().numpy()

            # flow_gt = batch['flow'][0]
            # gm0 = batch['gm0'][0]
            # flow_gt = flow_gt[~gm0]
            # flow_gt_np = flow_gt[mask0].clone().cpu().numpy()
            # flow_gt_np = flow_gt_np - flow_pose_np

            # scene_idx = batch['scene_id']
            # timestamp = batch['timestamp']
            # print(type(scene_idx), scene_idx)
            # # error = np.linalg.norm(flow_gt_np - flow_np, axis=1)
            # self.visualize(pc0_np, pc1_np, label0_np, label1_np, flow_np, flow_gt_np, png_name= f'visualizations/{scene_idx}_{timestamp}')
            # # exit()

            flow = torch.zeros((len(pc0), 3), device=pc0.device)
            flow[mask0] = flow_selected

        ret_dict = {}
        ret_dict["flow"] = [flow] # same size as original pcd wihtout removing invalid pts
        
        return ret_dict
      