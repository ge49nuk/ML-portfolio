import os
import torch
from tqdm import tqdm
from roitr_lib.trainer import Trainer
from roitr_lib.utils import to_o3d_pcd
from roitr_visualizer.visualizer import Visualizer, create_visualizer
from roitr_visualizer.feature_space import visualize_feature_space
import open3d as o3d
import numpy as np

class Tester(Trainer):
    '''
    Tester
    '''

    def __init__(self, config):
        Trainer.__init__(self, config)

    def test(self):
        print('Starting to evaluate on test datasets...')
        os.makedirs(f'{self.snapshot_dir}/{self.config.benchmark}', exist_ok=True)
        stats_meter = self.stats_meter()

        num_iter = len(self.loader['test'])

        c_loader_iter = self.loader['test'].__iter__()
        self.model.eval()
        with torch.no_grad():
            for idx in tqdm(range(num_iter)):
                torch.cuda.synchronize()
                inputs = next(c_loader_iter)

                if inputs["src_points"].shape[0] < 4096 or torch.sum(inputs["mask"]) < 4096:
                    self.skipped+=1
                    continue
            
                #######################################
                # Load inputs to device
                for k, v in inputs.items():
                    if v is None:
                        pass
                    elif type(v) == list:
                        inputs[k] = [items.to(self.device) for items in v]
                    else:
                        inputs[k] = v.to(self.device)
                ##################
                # forward pass
                ##################
                rot, trans = inputs['rot'][0], inputs['trans'][0]
                src_pcd, tgt_pcd = inputs['src_points'].contiguous(), inputs['tgt_points'].contiguous()
                src_normals, tgt_normals = inputs['src_normals'].contiguous(), inputs[
                    'tgt_normals'].contiguous()
                src_feats, tgt_feats = inputs['src_feats'].contiguous(), inputs['tgt_feats'].contiguous()
                src_raw_pcd = inputs['raw_src_pcd'].contiguous()
                obj_id = inputs['obj_id'][0].item()
                og_id = inputs['og_id'][0].item()


                outputs = self.model.forward(src_pcd, tgt_pcd, src_feats, tgt_feats, src_normals, tgt_normals,
                                             rot, trans, src_raw_pcd)

                data = dict()
                data['src_raw_pcd'] = src_raw_pcd.cpu()
                data['src_pcd'], data['tgt_pcd'] = src_pcd.cpu(), tgt_pcd.cpu()
                data['src_nodes'], data['tgt_nodes'] = outputs['src_nodes'].cpu(), outputs['tgt_nodes'].cpu()
                data['src_node_desc'], data['tgt_node_desc'] = outputs['src_node_feats'].cpu().detach(), outputs['tgt_node_feats'].cpu().detach()
                data['src_point_desc'], data['tgt_point_desc'] = outputs['src_point_feats'].cpu().detach(), outputs['tgt_point_feats'].cpu().detach()
                data['src_corr_pts'], data['tgt_corr_pts'] = outputs['src_corr_points'].cpu(), outputs['tgt_corr_points'].cpu()
                data['confidence'] = outputs['corr_scores'].cpu().detach()
                data['gt_tgt_node_occ'] = outputs['gt_tgt_node_occ'].cpu()
                data['gt_src_node_occ'] = outputs['gt_src_node_occ'].cpu()
                data['rot'], data['trans'] = rot.cpu(), trans.cpu()
                if self.config.benchmark == '4DMatch' or self.config.benchmark == '4DLoMatch':
                    data['metric_index_list'] = inputs['metric_index']
                    
                    
                ##################
                # Save outputs
                ##################
                sample_key = f"{torch.mean(rot).item() + og_id}"

                num_corrs = outputs["src_corr_points"].unique().shape[0]
                
                # Uncomment for GT correspondences
                
                # from scipy.spatial import cKDTree
                # tree = cKDTree(tgt_pcd.cpu().numpy())
                # gt_src = np.matmul(src_pcd.cpu().numpy() + trans.cpu().numpy().T, rot.cpu().numpy().T)
                # distances, indices = tree.query(gt_src, k=1)
                # close_points = distances < self.config.matching_radius
                # num_corrs = close_points.sum()
                
                if num_corrs == 0:
                    mean_conf = 0 
                else:
                    mean_conf = torch.mean(outputs["corr_scores"]).item()

                
                measure = num_corrs
                # Save the number of correlations, confidence and how often the correct object had the most corrs
                if obj_id != og_id:
                    self.corrs[1].append(num_corrs)
                    self.conf[1].append(mean_conf) 
                    measure *= -1
                else:
                    self.corrs[0].append(num_corrs)
                    self.conf[0].append(mean_conf) 
                
                if sample_key not in self.max_corrs_per_sample:
                    self.max_corrs_per_sample[sample_key] = measure
                elif abs(measure) > abs(self.max_corrs_per_sample[sample_key]):
                    self.max_corrs_per_sample[sample_key] = measure

                # torch.save(data, f'{self.snapshot_dir}/{self.config.benchmark}/{idx}.pth')
                
                stats = self.loss_func(outputs, inputs)
                evaluator_stats = self.evaluator(outputs, inputs)
                stats.update(evaluator_stats)
                
                if inputs['obj_id'][0].item() == inputs['og_id'][0].item():
                    for key, value in stats.items():
                        stats_meter[key].update(value)
                ###########################################################
        
        print("Skipped", self.skipped, "because of too small mask size")
        
        # Print results
        if self.local_rank <= 0:
            message = ''
            for key, value in stats_meter.items():
                message += f'{key}: {value.avg:.4f}\t'
        
        for k in [0,1]:
            if k == 0:
                print("********************* Correct objects *********************")
                pairs = stats_meter["IR"].avg * np.mean(self.corrs[k]).item()
                print(f"Mean correctly matched pairs:   {pairs}")
            else:
                print("********************** Wrong objects **********************")
            print(f"Support:                        {len(self.conf[k])}")
            print(f"Mean confidence:                {np.mean(self.conf[k])}")
            print(f"Median correspondences:         {np.median(self.corrs[k])}")
            print(f"Mean correspondences:           {np.mean(self.corrs[k])}")  
        
        num_correct = 0
        for key, val in self.max_corrs_per_sample.items():
            if val >= 0:
                num_correct += 1
        print(f"Correct object matched:         {num_correct}/{len(self.max_corrs_per_sample)} = {num_correct/len(self.max_corrs_per_sample)}")
        print()
        for key, value in stats_meter.items():
            print(f'{key}:{value.avg}') 

    





def get_trainer(config):
    '''
    Get corresponding trainer according to the config file
    :param config:
    :return:
    '''
    print(config.dataset)
    if config.dataset == 'tdmatch' or config.dataset == 'fdmatch' or config.dataset == 'bop':
        return Tester(config)
    else:
        raise NotImplementedError
