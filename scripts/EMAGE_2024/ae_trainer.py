import train
import os
import time
import csv
import sys
import warnings
import random
import numpy as np
import time
import pprint
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from loguru import logger
import smplx
from tqdm import tqdm,trange

from utils import config, logger_tools, other_tools, metric
from utils import rotation_conversions as rc
from dataloaders import data_tools
from optimizers.optim_factory import create_optimizer
from optimizers.scheduler_factory import create_scheduler
from optimizers.loss_factory import get_loss_func
from scipy.spatial.transform import Rotation


class CustomTrainer(train.BaseTrainer):
    """
    motion representation learning
    """
    def __init__(self, args):
        super().__init__(args)
        self.joints = self.train_data.joints
        self.smplx = smplx.create(
            self.args.data_path_1+"smplx_models/", 
            model_type='smplx',
            gender='NEUTRAL_2020', 
            use_face_contour=False,
            num_betas=300,
            num_expression_coeffs=100, 
            ext='npz',
            use_pca=False,
        ).cuda().eval()
        self.tracker = other_tools.EpochTracker(["rec", "vel", "ver", "ver_vel", "ver_acc", "com", "kl", "acc", "loss"], 
                                                [False, False, False, False, False, False, False, False, False])
        if not self.args.rot6d: #"rot6d" not in args.pose_rep:
            logger.error(f"this script is for rot6d, your pose rep. is {args.pose_rep}")
        self.rec_loss = get_loss_func("GeodesicLoss")
        self.vel_loss = torch.nn.L1Loss(reduction='mean')
        self.vectices_loss = torch.nn.MSELoss(reduction='mean')
    
    def inverse_selection(self, filtered_t, selection_array, n):
        # 创建一个全为零的数组，形状为 n*165
        original_shape_t = np.zeros((n, selection_array.size))
        
        # 找到选择数组中为1的索引位置
        selected_indices = np.where(selection_array == 1)[0]
        
        # 将 filtered_t 的值填充到 original_shape_t 中相应的位置
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
            
        return original_shape_t
    
    def inverse_selection_tensor(self, filtered_t, selection_array, n):
    # 创建一个全为零的数组，形状为 n*165
        selection_array = torch.from_numpy(selection_array).cuda()
        original_shape_t = torch.zeros((n, 165)).cuda()
        
        # 找到选择数组中为1的索引位置
        selected_indices = torch.where(selection_array == 1)[0]
        
        # 将 filtered_t 的值填充到 original_shape_t 中相应的位置
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
            
        return original_shape_t

    def train(self, epoch):
        self.model.train()
        t_start = time.time()
        for its, dict_data in enumerate(tqdm(self.train_loader,desc=f'train epoch {epoch}')):
            self.tracker.reset()
            tar_pose = dict_data["pose"]
            tar_beta = dict_data["beta"].cuda()
            tar_trans = dict_data["trans"].cuda()
            tar_pose = tar_pose.cuda()  
            bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints
            tar_exps = torch.zeros((bs, n, 100)).cuda()
            tar_pose = rc.axis_angle_to_matrix(tar_pose.reshape(bs, n, j, 3))
            tar_pose = rc.matrix_to_rotation_6d(tar_pose).reshape(bs, n, j*6)
            t_data = time.time() - t_start 
            
            self.opt.zero_grad()
            g_loss_final = 0
            net_out = self.model(tar_pose)
            rec_pose = net_out["rec_pose"]
            rec_pose = rec_pose.reshape(bs, n, j, 6)
            rec_pose = rc.rotation_6d_to_matrix(rec_pose)#
            tar_pose = rc.rotation_6d_to_matrix(tar_pose.reshape(bs, n, j, 6))
            loss_rec = self.rec_loss(rec_pose, tar_pose)
            self.tracker.update_meter("rec", "train", loss_rec.item())
            g_loss_final += loss_rec * self.args.rec_weight

            velocity_loss =  self.vel_loss(rec_pose[:, 1:] - rec_pose[:, :-1], tar_pose[:, 1:] - tar_pose[:, :-1])
            acceleration_loss =  self.vel_loss(rec_pose[:, 2:] + rec_pose[:, :-2] - 2 * rec_pose[:, 1:-1], tar_pose[:, 2:] + tar_pose[:, :-2] - 2 * tar_pose[:, 1:-1])
            self.tracker.update_meter("vel", "train", velocity_loss.item())
            self.tracker.update_meter("acc", "train", acceleration_loss.item())
            g_loss_final += velocity_loss * self.args.rec_vel_weight
            g_loss_final += acceleration_loss * self.args.rec_acc_weight

             # vertices loss
            tar_pose = rc.matrix_to_axis_angle(tar_pose).reshape(bs*n, j*3)
            rec_pose = rc.matrix_to_axis_angle(rec_pose).reshape(bs*n, j*3)
            rec_pose = self.inverse_selection_tensor(rec_pose, self.train_data.joint_mask, rec_pose.shape[0])
            tar_pose = self.inverse_selection_tensor(tar_pose, self.train_data.joint_mask, tar_pose.shape[0])
            vertices_rec = self.smplx(
                betas=tar_beta.reshape(bs*n, 300), 
                transl=tar_trans.reshape(bs*n, 3), 
                expression=tar_exps.reshape(bs*n, 100), 
                jaw_pose=rec_pose[:, 66:69], 
                global_orient=rec_pose[:,:3], 
                body_pose=rec_pose[:,3:21*3+3], 
                left_hand_pose=rec_pose[:,25*3:40*3], 
                right_hand_pose=rec_pose[:,40*3:55*3], 
                return_verts=True,
                return_joints=True,
                leye_pose=tar_pose[:, 69:72], 
                reye_pose=tar_pose[:, 72:75],
            )
            vertices_tar = self.smplx(
                betas=tar_beta.reshape(bs*n, 300), 
                transl=tar_trans.reshape(bs*n, 3), 
                expression=tar_exps.reshape(bs*n, 100), 
                jaw_pose=tar_pose[:, 66:69], 
                global_orient=tar_pose[:,:3], 
                body_pose=tar_pose[:,3:21*3+3], 
                left_hand_pose=tar_pose[:,25*3:40*3], 
                right_hand_pose=tar_pose[:,40*3:55*3], 
                return_verts=True,
                return_joints=True,
                leye_pose=tar_pose[:, 69:72], 
                reye_pose=tar_pose[:, 72:75],
            )  
            vectices_loss = self.vectices_loss(vertices_rec['vertices'], vertices_tar['vertices'])
            self.tracker.update_meter("ver", "train", vectices_loss.item())
            g_loss_final += vectices_loss * self.args.rec_ver_weight

            vertices_vel_loss = self.vel_loss(vertices_rec['vertices'][:, 1:] - vertices_rec['vertices'][:, :-1], vertices_tar['vertices'][:, 1:] - vertices_tar['vertices'][:, :-1])
            vertices_acc_loss = self.vel_loss(vertices_rec['vertices'][:, 2:] + vertices_rec['vertices'][:, :-2] - 2 * vertices_rec['vertices'][:, 1:-1], vertices_tar['vertices'][:, 2:] + vertices_tar['vertices'][:, :-2] - 2 * vertices_tar['vertices'][:, 1:-1])
            self.tracker.update_meter("ver_vel", "train", vertices_vel_loss.item())
            self.tracker.update_meter("ver_acc", "train", vertices_acc_loss.item())
            g_loss_final += vertices_vel_loss * self.args.rec_ver_vel_weight
            g_loss_final += vertices_acc_loss * self.args.rec_ver_acc_weight
            
            loss_embedding = net_out["embedding_loss"]
            self.tracker.update_meter("com", "train", loss_embedding.item())
            g_loss_final += loss_embedding * self.args.comm_weight

            self.tracker.update_meter("loss", "train", g_loss_final.item())
            g_loss_final.backward()
            if self.args.grad_norm > 0: 
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_norm)
            self.opt.step()
            t_train = time.time() - t_start - t_data
            t_start = time.time()
            mem_cost = torch.cuda.memory_cached() / 1E9
            lr_g = self.opt.param_groups[0]['lr']
            if its % self.args.log_period == 0:
                self.train_recording(epoch, its, t_data, t_train, mem_cost, lr_g)   
            
            # Save checkpoint and run val 4 times per training epoch.
            if its % (len(self.train_loader)/3) < 1:
                other_tools.save_checkpoints(os.path.join(self.checkpoint_path, 
                                                          f"epoch_{epoch:04}_iter_{its:05}.bin"), 
                                                          self.model, 
                                                          opt=None, 
                                                          epoch=None, 
                                                          lrs=None)
                self.val(epoch,its)
                self.model.train()

        self.opt_s.step(epoch)
                    
    def val(self, epoch, train_its):
        self.model.eval()
        t_start = time.time()
        with torch.no_grad():
            for its, dict_data in enumerate(tqdm(self.val_loader,desc=f'val epoch {epoch} iter {train_its}')):
                tar_pose = dict_data["pose"]
                tar_beta = dict_data["beta"].cuda()
                tar_trans = dict_data["trans"].cuda()
                tar_pose = tar_pose.cuda()  
                bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints
                tar_exps = torch.zeros((bs, n, 100)).cuda()
                tar_pose = rc.axis_angle_to_matrix(tar_pose.reshape(bs, n, j, 3))
                tar_pose = rc.matrix_to_rotation_6d(tar_pose).reshape(bs, n, j*6)
                t_data = time.time() - t_start 

                g_loss_final = 0
                net_out = self.model(tar_pose)
                rec_pose = net_out["rec_pose"]
                rec_pose = rec_pose.reshape(bs, n, j, 6)
                rec_pose = rc.rotation_6d_to_matrix(rec_pose)#
                tar_pose = rc.rotation_6d_to_matrix(tar_pose.reshape(bs, n, j, 6))
                loss_rec = self.rec_loss(rec_pose, tar_pose)
                self.tracker.update_meter("rec", "val", loss_rec.item())
                g_loss_final += loss_rec * self.args.rec_weight

                velocity_loss =  self.vel_loss(rec_pose[:, 1:] - rec_pose[:, :-1], tar_pose[:, 1:] - tar_pose[:, :-1])
                acceleration_loss =  self.vel_loss(rec_pose[:, 2:] + rec_pose[:, :-2] - 2 * rec_pose[:, 1:-1], tar_pose[:, 2:] + tar_pose[:, :-2] - 2 * tar_pose[:, 1:-1])
                self.tracker.update_meter("vel", "val", velocity_loss.item())
                self.tracker.update_meter("acc", "val", acceleration_loss.item())
                g_loss_final += velocity_loss * self.args.rec_vel_weight
                g_loss_final += acceleration_loss * self.args.rec_acc_weight

                tar_pose = rc.matrix_to_axis_angle(tar_pose).reshape(bs*n, j*3)
                rec_pose = rc.matrix_to_axis_angle(rec_pose).reshape(bs*n, j*3)
                rec_pose = self.inverse_selection_tensor(rec_pose, self.train_data.joint_mask, rec_pose.shape[0])
                tar_pose = self.inverse_selection_tensor(tar_pose, self.train_data.joint_mask, tar_pose.shape[0])
                vertices_rec = self.smplx(
                    betas=tar_beta.reshape(bs*n, 300), 
                    transl=tar_trans.reshape(bs*n, 3), 
                    expression=tar_exps.reshape(bs*n, 100), 
                    jaw_pose=rec_pose[:, 66:69], 
                    global_orient=rec_pose[:,:3], 
                    body_pose=rec_pose[:,3:21*3+3], 
                    left_hand_pose=rec_pose[:,25*3:40*3], 
                    right_hand_pose=rec_pose[:,40*3:55*3], 
                    return_verts=True, 
                    leye_pose=tar_pose[:, 69:72], 
                    reye_pose=tar_pose[:, 72:75],
                )
                vertices_tar = self.smplx(
                    betas=tar_beta.reshape(bs*n, 300), 
                    transl=tar_trans.reshape(bs*n, 3), 
                    expression=tar_exps.reshape(bs*n, 100), 
                    jaw_pose=tar_pose[:, 66:69], 
                    global_orient=tar_pose[:,:3], 
                    body_pose=tar_pose[:,3:21*3+3], 
                    left_hand_pose=tar_pose[:,25*3:40*3], 
                    right_hand_pose=tar_pose[:,40*3:55*3], 
                    return_verts=True, 
                    leye_pose=tar_pose[:, 69:72], 
                    reye_pose=tar_pose[:, 72:75],
                )  
                vectices_loss = self.vectices_loss(vertices_rec['vertices'], vertices_tar['vertices'])
                self.tracker.update_meter("ver", "val", vectices_loss.item())
                g_loss_final += vectices_loss * self.args.rec_ver_weight

                vertices_vel_loss = self.vel_loss(vertices_rec['vertices'][:, 1:] - vertices_rec['vertices'][:, :-1], vertices_tar['vertices'][:, 1:] - vertices_tar['vertices'][:, :-1])
                vertices_acc_loss = self.vel_loss(vertices_rec['vertices'][:, 2:] + vertices_rec['vertices'][:, :-2] - 2 * vertices_rec['vertices'][:, 1:-1], vertices_tar['vertices'][:, 2:] + vertices_tar['vertices'][:, :-2] - 2 * vertices_tar['vertices'][:, 1:-1])
                self.tracker.update_meter("ver_vel", "val", vertices_vel_loss.item())
                self.tracker.update_meter("ver_acc", "val", vertices_acc_loss.item())
                g_loss_final += vertices_vel_loss * self.args.rec_ver_vel_weight
                g_loss_final += vertices_acc_loss * self.args.rec_ver_acc_weight

                loss_embedding = net_out["embedding_loss"]
                self.tracker.update_meter("com", "val", loss_embedding.item())
                g_loss_final += loss_embedding * self.args.comm_weight

                self.tracker.update_meter("loss", "val", g_loss_final.item())
            self.val_recording(epoch,train_its)

    def test(self, epoch):
        start_time = time.time()
        total_length = 0
        test_seq_list = self.test_data.selected_file
        self.model.eval()
        from tqdm import tqdm,trange
        count = {'2_scott':0,
                 '21_ayana':0}
        max_seq_len = 900 
        max_num_examples = 8
        with torch.no_grad():
            for its, dict_data in tqdm(enumerate(self.test_loader),desc='examples',total=len(self.test_loader)):
                for speaker in count:
                    if (speaker in test_seq_list.iloc[its]['id'] and
                        count[speaker] < max_num_examples):
                        count[speaker] += 1
                        break
                else:
                    if 'beat' in test_seq_list.iloc[its]['id']:
                        continue
                    
                tar_pose = dict_data["pose"][:,:max_seq_len].cuda()
                bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints
                tar_pose = rc.axis_angle_to_matrix(tar_pose.reshape(bs,n,j,3))
                tar_pose = rc.matrix_to_rotation_6d(tar_pose).reshape(bs,n,j*6)
                ################################################################
                gt_npz = np.load(self.args.data_path+self.args.pose_rep+"/"+test_seq_list.iloc[its]['id']+'.npz', allow_pickle=True)
                assert dict_data['pose'].shape[1] <= gt_npz['poses'].shape[0]
                tar_betas = gt_npz['betas'].astype(np.float32)
                tar_betas = np.pad(tar_betas,(0,300-tar_betas.shape[0]),'constant',constant_values=0)
                tar_betas = torch.from_numpy(np.repeat(tar_betas.reshape(1,300),n,0)).cuda()
                tar_trans = torch.from_numpy(gt_npz['trans'][:n].astype(np.float32)).cuda() * 0
                tar_exps = gt_npz.get('expressions',np.zeros((n,100),dtype=np.float32))
                tar_exps = torch.from_numpy(tar_exps[:n].astype(np.float32)).cuda() * 0
                ################################################################

                net_out = self.model(tar_pose)
                rec_pose = net_out["rec_pose"]
                n = rec_pose.shape[1]
                tar_pose = tar_pose[:, :n, :]
                rec_pose = rec_pose.reshape(bs, n, j, 6) 
                rec_pose = rc.rotation_6d_to_matrix(rec_pose)#
                rec_pose = rc.matrix_to_axis_angle(rec_pose).reshape(bs*n,j*3)
                            
                tar_pose = rc.rotation_6d_to_matrix(tar_pose.reshape(bs,n,j,6))
                tar_pose = rc.matrix_to_axis_angle(tar_pose).reshape(bs*n,j*3)
                
                total_length += n 

                import trimesh
                assert bs == 1
                tar_pose = self.inverse_selection_tensor(tar_pose, self.test_data.joint_mask, tar_pose.shape[0])
                rec_pose = self.inverse_selection_tensor(rec_pose, self.test_data.joint_mask, rec_pose.shape[0])

                pkl_f = 'meshes/' + test_seq_list.iloc[its]['id'] + '.pkl'
                os.makedirs('meshes',exist_ok=True)
                if 'lower' in self.args.wandb_run:
                    part = 'lower'
                elif 'upper' in self.args.wandb_run:
                    part = 'upper'
                elif 'hands' in self.args.wandb_run:
                    part = 'hands'
                write_mesh = False
                if not os.path.exists(pkl_f):
                    pickle.dump({'tar_beta':tar_betas.detach().cpu().numpy(),
                                 'tar_trans':tar_trans.detach().cpu().numpy(),
                                 'tar_exps':tar_exps.detach().cpu().numpy(),
                                 'tar_pose':tar_pose.detach().cpu().numpy(),
                                 'rec_pose':rec_pose.detach().cpu().numpy(),
                                 'part':[part]},
                                open(pkl_f,'wb'))
                else:
                    pkl = pickle.load(open(pkl_f,'rb'))
                    inds = np.where(self.test_data.joint_mask == 1)[0]
                    pkl['rec_pose'][:,inds] = rec_pose[:,inds].detach().cpu().numpy()
                    pkl['tar_pose'][:,inds] = tar_pose[:,inds].detach().cpu().numpy()
                    pkl['part'].append(part)
                    pickle.dump(pkl,open(pkl_f,'wb'))
                    rec_pose = torch.from_numpy(pkl['rec_pose']).cuda()
                    tar_pose = torch.from_numpy(pkl['tar_pose']).cuda()
                    write_mesh = (set(pkl['part']) == {'lower','upper','hands'})
                if not write_mesh:
                    continue
                
                body = self.smplx(
                    betas=tar_betas.reshape(bs*n, 300), 
                    transl=tar_trans.reshape(bs*n, 3), 
                    expression=tar_exps.reshape(bs*n, 100), 
                    jaw_pose=rec_pose[:, 66:69], 
                    global_orient=rec_pose[:,:3], 
                    body_pose=rec_pose[:,3:21*3+3], 
                    left_hand_pose=rec_pose[:,25*3:40*3], 
                    right_hand_pose=rec_pose[:,40*3:55*3], 
                    return_verts=True, 
                    leye_pose=rec_pose[:, 69:72], 
                    reye_pose=rec_pose[:, 72:75],
                )
                vertices = body.vertices.detach().cpu().numpy()
                ply_d = 'meshes/' + test_seq_list.iloc[its]['id']
                os.makedirs(ply_d,exist_ok=True)
                for i in trange(min(n,max_seq_len),desc='writing rec meshes',leave=False):
                    ply_f = f'{ply_d}/{i:05}.ply'
                    mesh = trimesh.Trimesh(vertices=vertices[i],
                                            faces=self.smplx.faces,
                                            process=False)
                    mesh.export(ply_f)
                body = self.smplx(
                    betas=tar_betas.reshape(bs*n, 300), 
                    transl=tar_trans.reshape(bs*n, 3), 
                    expression=tar_exps.reshape(bs*n, 100), 
                    jaw_pose=tar_pose[:, 66:69], 
                    global_orient=tar_pose[:,:3], 
                    body_pose=tar_pose[:,3:21*3+3], 
                    left_hand_pose=tar_pose[:,25*3:40*3], 
                    right_hand_pose=tar_pose[:,40*3:55*3], 
                    return_verts=True, 
                    leye_pose=tar_pose[:, 69:72], 
                    reye_pose=tar_pose[:, 72:75],
                )
                vertices = body.vertices.detach().cpu().numpy()
                ply_d = 'meshes/' + test_seq_list.iloc[its]['id'] + '_gt'
                os.makedirs(ply_d,exist_ok=True)
                for i in trange(min(n,max_seq_len),desc='writing tar meshes',leave=False):
                    ply_f = f'{ply_d}/{i:05}.ply'
                    mesh = trimesh.Trimesh(vertices=vertices[i],
                                            faces=self.smplx.faces,
                                            process=False)
                    mesh.export(ply_f)                                                                                                
        end_time = time.time() - start_time
        logger.info(f"total inference time: {int(end_time)} s for {int(total_length/self.args.pose_fps)} s motion")


    def test_losses(self):
        self.model.eval()
        losses = dict()
        with torch.no_grad():
            for its, dict_data in enumerate(tqdm(self.test_loader,desc=f'examples')):
                losses_rec = list()
                losses_ver = list()
                max_n_frames = 4096
                n_frames = dict_data["pose"].shape[1]
                for i_beg in range(0,n_frames,max_n_frames):
                    i_end = min(i_beg + max_n_frames,n_frames)
                    id = self.test_data.selected_file.iloc[its]['id']

                    tar_pose = dict_data["pose"][:,i_beg:i_end].cuda()
                    tar_beta = dict_data["beta"][:,i_beg:i_end].cuda()
                    tar_trans = dict_data["trans"][:,i_beg:i_end].cuda()
                    bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints
                    tar_exps = torch.zeros((bs, n, 100)).cuda()
                    tar_pose = rc.axis_angle_to_matrix(tar_pose.reshape(bs, n, j, 3))
                    tar_pose = rc.matrix_to_rotation_6d(tar_pose).reshape(bs, n, j*6)

                    g_loss_final = 0
                    net_out = self.model(tar_pose)
                    rec_pose = net_out["rec_pose"]
                    rec_pose = rec_pose.reshape(bs, n, j, 6)
                    rec_pose = rc.rotation_6d_to_matrix(rec_pose)#
                    tar_pose = rc.rotation_6d_to_matrix(tar_pose.reshape(bs, n, j, 6))
                    loss_rec = self.rec_loss(rec_pose, tar_pose)
                    g_loss_final += loss_rec * self.args.rec_weight

                    velocity_loss =  self.vel_loss(rec_pose[:, 1:] - rec_pose[:, :-1], tar_pose[:, 1:] - tar_pose[:, :-1])
                    acceleration_loss =  self.vel_loss(rec_pose[:, 2:] + rec_pose[:, :-2] - 2 * rec_pose[:, 1:-1], tar_pose[:, 2:] + tar_pose[:, :-2] - 2 * tar_pose[:, 1:-1])
                    g_loss_final += velocity_loss * self.args.rec_vel_weight
                    g_loss_final += acceleration_loss * self.args.rec_acc_weight

                    tar_pose = rc.matrix_to_axis_angle(tar_pose).reshape(bs*n, j*3)
                    rec_pose = rc.matrix_to_axis_angle(rec_pose).reshape(bs*n, j*3)
                    rec_pose = self.inverse_selection_tensor(rec_pose, self.train_data.joint_mask, rec_pose.shape[0])
                    tar_pose = self.inverse_selection_tensor(tar_pose, self.train_data.joint_mask, tar_pose.shape[0])
                    vertices_rec = self.smplx(
                        betas=tar_beta.reshape(bs*n, 300), 
                        transl=tar_trans.reshape(bs*n, 3), 
                        expression=tar_exps.reshape(bs*n, 100), 
                        jaw_pose=rec_pose[:, 66:69], 
                        global_orient=rec_pose[:,:3], 
                        body_pose=rec_pose[:,3:21*3+3], 
                        left_hand_pose=rec_pose[:,25*3:40*3], 
                        right_hand_pose=rec_pose[:,40*3:55*3], 
                        return_verts=True, 
                        leye_pose=tar_pose[:, 69:72], 
                        reye_pose=tar_pose[:, 72:75],
                    )
                    vertices_tar = self.smplx(
                        betas=tar_beta.reshape(bs*n, 300), 
                        transl=tar_trans.reshape(bs*n, 3), 
                        expression=tar_exps.reshape(bs*n, 100), 
                        jaw_pose=tar_pose[:, 66:69], 
                        global_orient=tar_pose[:,:3], 
                        body_pose=tar_pose[:,3:21*3+3], 
                        left_hand_pose=tar_pose[:,25*3:40*3], 
                        right_hand_pose=tar_pose[:,40*3:55*3], 
                        return_verts=True, 
                        leye_pose=tar_pose[:, 69:72], 
                        reye_pose=tar_pose[:, 72:75],
                    )  
                    vertices_loss = self.vectices_loss(vertices_rec['vertices'], vertices_tar['vertices'])
                    g_loss_final += vertices_loss * self.args.rec_ver_weight

                    vertices_vel_loss = self.vel_loss(vertices_rec['vertices'][:, 1:] - vertices_rec['vertices'][:, :-1], vertices_tar['vertices'][:, 1:] - vertices_tar['vertices'][:, :-1])
                    vertices_acc_loss = self.vel_loss(vertices_rec['vertices'][:, 2:] + vertices_rec['vertices'][:, :-2] - 2 * vertices_rec['vertices'][:, 1:-1], vertices_tar['vertices'][:, 2:] + vertices_tar['vertices'][:, :-2] - 2 * vertices_tar['vertices'][:, 1:-1])
                    g_loss_final += vertices_vel_loss * self.args.rec_ver_vel_weight
                    g_loss_final += vertices_acc_loss * self.args.rec_ver_acc_weight

                    loss_embedding = net_out["embedding_loss"]
                    g_loss_final += loss_embedding * self.args.comm_weight

                    losses_rec.append(loss_rec.detach().cpu().numpy())
                    losses_ver.append(vertices_loss.detach().cpu().numpy())
                losses[id] = {'rec':np.mean(losses_rec).item(),
                              'ver':np.mean(losses_ver).item()}
        return losses


    def test_gt(self, epoch):
        test_seq_list = self.test_data.selected_file
        self.model.eval()
        from tqdm import tqdm,trange
        max_seq_len = 30*30
        speakers = set()
        with torch.no_grad():
            for its, dict_data in tqdm(enumerate(self.test_loader),desc='examples',total=len(self.test_loader)):
                # i1 = test_seq_list.iloc[its]['id'].rfind('_')
                # i2 = test_seq_list.iloc[its]['id'].rfind('_',0,i1)
                # i3 = test_seq_list.iloc[its]['id'].rfind('_',0,i2)
                # speaker = test_seq_list.iloc[its]['id'][:i3]
                # if speaker in speakers:
                #     continue
                # speakers.add(speaker)

                tar_pose = dict_data["pose"][:,:max_seq_len]
                tar_pose = tar_pose.cuda()
                bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints
                ################################################################
                gt_npz = np.load(self.args.data_path+self.args.pose_rep+"/"+test_seq_list.iloc[its]['id']+'.npz', allow_pickle=True)
                assert dict_data['pose'].shape[1] <= gt_npz['poses'].shape[0]
                tar_beta = gt_npz['betas'].astype(np.float32)
                tar_beta = np.pad(tar_beta,(0,300-tar_beta.shape[0]),'constant',constant_values=0)
                tar_beta = torch.from_numpy(np.repeat(tar_beta.reshape(1,300),n,0)).cuda()
                tar_trans = torch.from_numpy(gt_npz['trans'][:n].astype(np.float32)).cuda()
                if 'expressions' in gt_npz:
                    tar_exps = torch.from_numpy(gt_npz['expressions'][:n].astype(np.float32)).cuda()
                else:
                    tar_exps = torch.zeros((n,100),dtype=torch.float32).cuda()
                tar_pose = torch.from_numpy(gt_npz['poses'][:n].astype(np.float32)).cuda()
                ################################################################

                import trimesh
                assert bs == 1

                body = self.smplx(
                    betas=tar_beta.reshape(bs*n, 300), 
                    transl=tar_trans.reshape(bs*n, 3), 
                    expression=tar_exps.reshape(bs*n, 100), 
                    jaw_pose=tar_pose[:, 66:69], 
                    global_orient=tar_pose[:,:3], 
                    body_pose=tar_pose[:,3:21*3+3], 
                    left_hand_pose=tar_pose[:,25*3:40*3], 
                    right_hand_pose=tar_pose[:,40*3:55*3], 
                    return_verts=True, 
                    leye_pose=tar_pose[:, 69:72], 
                    reye_pose=tar_pose[:, 72:75],
                )
                vertices = body.vertices.detach().cpu().numpy()
                ply_d = 'meshes/' + test_seq_list.iloc[its]['id'] + '_gt'
                os.makedirs(ply_d,exist_ok=True)
                for i in trange(min(n,max_seq_len),desc='writing tar meshes',leave=False):
                    ply_f = f'{ply_d}/{i:05}.ply'
                    mesh = trimesh.Trimesh(vertices=vertices[i],
                                            faces=self.smplx.faces,
                                            process=False)
                    mesh.export(ply_f)                                                                                                
