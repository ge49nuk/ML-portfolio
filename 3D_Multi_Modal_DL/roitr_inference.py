import torch

from roitr_model.RIGA_v2 import create_model
from easydict import EasyDict as edict
from configs.utils import load_config
from roitr_lib.utils import calc_pc_radius, to_ply_file, remove_outliers, to_o3d_pcd
from roitr_dataset.common import normal_redirect
import numpy as np
import open3d as o3d

 
def inference(model, inputs, device="cuda:0"):

    model.eval()
    with torch.no_grad():
        rot, trans = torch.zeros((3,3)), torch.zeros((3,1))
        src_feats = torch.ones(size=(inputs['src_points'].shape[0], 1)).to(device)
        tgt_feats = torch.ones(size=(inputs['tgt_points'].shape[0], 1)).to(device)
        src_pcd, tgt_pcd = inputs['src_points'], inputs['tgt_points']
        if 'trans' in inputs:
            trans = inputs['trans'].float().cpu()
        if 'rot' in inputs:
            rot = inputs['rot'].float().cpu()
        
        # normalize point clouds
        center_tgt = torch.mean(tgt_pcd, axis=0)
        tgt_pcd -= center_tgt
        rad_tgt = calc_pc_radius(tgt_pcd)

        src_pcd, center_src, src_samples = remove_outliers(src_pcd.numpy(), rad_tgt)
        src_pcd = torch.from_numpy(src_pcd) - center_src
        trans += center_src.reshape(3,1)
        src_pcd /= rad_tgt
        trans /= rad_tgt

        if src_pcd.shape[0] > 8192:
            idx = np.random.permutation(src_pcd.shape[0])[:8192]
            src_pcd = src_pcd[idx]
        elif src_pcd.shape[0] < 2048:
            return torch.zeros((0,3)), torch.zeros((0,3)), torch.zeros((0,3))
        
        # Normal estimation
        o3d_src_pcd = to_o3d_pcd(src_pcd.numpy())
        o3d_tgt_pcd = to_o3d_pcd(tgt_pcd.numpy())
        o3d_src_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=33))
        src_normals = np.asarray(o3d_src_pcd.normals)
        src_normals = normal_redirect(src_pcd.numpy(), src_normals, view_point=np.array((0,0,0)))
        o3d_tgt_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=33))
        tgt_normals = np.asarray(o3d_tgt_pcd.normals)
        tgt_normals = normal_redirect(tgt_pcd.numpy(), tgt_normals, view_point=np.array((0,0,0)))
        src_feats = np.ones(shape=(src_pcd.shape[0], 1))
        tgt_feats = np.ones(shape=(tgt_pcd.shape[0], 1))

        src_raw_pcd =  src_pcd.detach().clone().to(torch.float32).to(device)
        gt_src = torch.matmul(src_pcd + trans.T, rot.T)
        
        src_normals = torch.from_numpy(src_normals).to(torch.float32).to(device)
        tgt_normals = torch.from_numpy(tgt_normals).to(torch.float32).to(device)
        src_feats = torch.from_numpy(src_feats).to(torch.float32).to(device)
        tgt_feats = torch.from_numpy(tgt_feats).to(torch.float32).to(device)

        # Make contiguous in memory
        src_pcd, tgt_pcd = src_pcd.to(torch.float32).to(device).contiguous(), tgt_pcd.to(torch.float32).to(device).contiguous()
        src_normals, tgt_normals = src_normals.contiguous(), tgt_normals.contiguous()
        src_raw_pcd = src_raw_pcd.contiguous()

        # Move tensors to gpu
        for k, v in inputs.items():
            if type(v) == list:
                inputs[k] = [item.to(device) for item in v]
            elif v is None or type(v) == int or type(v) == float:
                inputs[k] = v
            else:
                inputs[k] = v.to(device)

        outputs = model.forward(src_pcd, tgt_pcd, src_feats, tgt_feats, src_normals, tgt_normals, rot, trans, src_raw_pcd, True)
        
        return outputs["corr_scores"], outputs["src_corr_points"], outputs["tgt_corr_points"]

def main():
    config_path = "configs/roitr/inference.yaml"
    config = load_config(config_path)
    config = edict(config)
    
    model = create_model(config).to("cuda:0") 
    state = torch.load(config.pretrain)
    model.load_state_dict(state['state_dict'])

    src_pts = torch.rand((8192,3))
    inputs = {
        'src_points': src_pts,
        'src_normals': torch.rand((8192,3)),
        'tgt_points': torch.rand((8192,3)),
        'raw_src_pcd': src_pts,
    }
    print(inference(model, inputs))

if __name__ == '__main__':
    main()

