import os, gc
import torch
from tensorboardX import SummaryWriter
from roitr_lib.utils import AverageMeter, Logger, to_o3d_pcd, to_ply_file
from tqdm import tqdm
import numpy as np
import open3d as o3d

class Trainer(object):
    '''
    Class Trainer
    '''

    def __init__(self, config):
        self.config = config
        #############################
        # hyper-parameters
        #############################
        self.verbose = config.verbose
        self.verbose_freq = config.verbose_freq
        self.start_epoch = 1
        self.max_epoch = config.max_epoch
        self.training_max_iter = config.training_max_iter
        self.val_max_iter = config.val_max_iter
        self.device = config.device

        self.best_loss = self.best_c_loss = self.best_f_loss = self.best_o_loss = self.best_train_loss = 1e5
        self.best_PIR = self.best_IR = -1.

        self.save_dir = config.save_dir
        self.snapshot_dir = config.snapshot_dir

        self.model = config.model.to(self.device)
        self.local_rank = config.local_rank
        self.optimizer = config.optimizer
        self.scheduler = config.scheduler
        self.scheduler_interval = config.scheduler_interval
        self.snapshot_interval = config.snapshot_interval
        self.iter_size = config.iter_size

        if self.local_rank <= 0:
            self.writer = SummaryWriter(logdir=config.tboard_dir)
            self.logger = Logger(self.snapshot_dir)
            self.logger.write(f'#parameters {sum([x.nelement() for x in self.model.parameters()]) / 1000000.} M\n')
            with open(f'{config.snapshot_dir}/model.log', 'w') as f:
                f.write(str(self.model))

            f.close()
        else:
            self.writer = None
            self.logger = None

        if config.pretrain != '':
            self._load_pretrain(config.pretrain)

        self.loader = dict()

        self.loader['train'] = config.train_loader
        self.loader['val'] = config.val_loader
        self.loader['test'] = config.test_loader

        self.loss_func = config.loss_func
        self.evaluator = config.evaluator
        
        self.conf = [[],[]]
        self.corrs = [[],[]]
        self.max_corrs_per_sample = {}
        self.skipped = 0

    def _snapshot(self, epoch, name=None):
        '''
        Save a trained model
        :param epoch:  epoch of current model
        :param name: path to save current model
        :return: None
        '''
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'best_c_loss': self.best_c_loss,
            'best_f_loss': self.best_f_loss,
            'best_o_loss': self.best_o_loss,
            'best_PIR': self.best_PIR,
            'best_IR': self.best_IR,
        }

        if name is None:
            filename = os.path.join(self.save_dir, f'model_{epoch}.pth')
        else:
            filename = os.path.join(self.save_dir, f'model_{name}.pth')

        print(f'Save model to {filename}')
        self.logger.write(f'Save model to {filename}\n')
        torch.save(state, filename)

    def _load_pretrain(self, resume):
        '''
        Load a pretrained model
        :param resume: the path to the pretrained model
        :return: None
        '''
        if os.path.isfile(resume):
            print(f'=> loading checkpoint {resume}')
            state = torch.load(resume)
            self.start_epoch = state['epoch']
            if self.local_rank < 0:
                #print(state['state_dict'].keys())
                self.model.load_state_dict({k.replace('module.', ''): v for k, v in state['state_dict'].items()})
                self.optimizer.load_state_dict(state['optimizer'])
                self.scheduler.load_state_dict(state['scheduler'])
            else:
                self.model.load_state_dict(state['state_dict'])
                self.optimizer.load_state_dict(state['optimizer'])
                self.scheduler.load_state_dict(state['scheduler'])

            if self.local_rank <= 0:
                self.best_loss = state['best_loss']
                self.best_c_loss = state['best_c_loss']
                self.best_f_loss = state['best_f_loss']
                self.best_o_loss = state['best_o_loss']
                self.best_PIR = state['best_PIR']
                self.best_IR = state['best_IR']
                self.logger.write(f'Successfully load pretrained model from {resume}!\n')
                self.logger.write(f'Current best loss {self.best_loss}\n')
                self.logger.write(f'Current best c_loss {self.best_c_loss}\n')
                self.logger.write(f'Current best f_loss {self.best_f_loss}\n')

                self.logger.write(f'Current best PIR {self.best_PIR}\n')
                self.logger.write(f'Current best IR {self.best_IR}\n')

        else:
            raise ValueError(f'=> no checkpoint found at {resume}')

    def _get_lr(self, group=0):
        '''
        Get current learning rate
        :param group:
        :return:
        '''
        return self.optimizer.param_groups[group]['lr']

    def stats_dict(self):
        '''
        Create the dictionary consisting of all the metrics
        :return: as described
        '''
        stats = dict()
        stats['loss'] = 0.
        stats['c_loss'] = 0.
        stats['f_loss'] = 0.
        stats['o_loss'] = 0.
        stats['PIR'] = 0.
        stats['IR'] = 0.

        '''
        to be added
        '''
        return stats

    def stats_meter(self):
        '''
        For each metric in stats dict, create an AverageMeter class for updating
        :return: as described
        '''
        meters = dict()
        stats = self.stats_dict()
        for key, _ in stats.items():
            meters[key] = AverageMeter()
        return meters

    def inference_one_batch(self, inputs, phase, idx=None):
        '''
        Inference for a single batch data
        :param inputs: the dictionary consisting of all the input data
        :param phase: train, validation or test
        :return: dictionary consisting of losses and metrics
        '''
        assert phase in ['train', 'val']
        #########################################
        # training
        #########################################
        if phase == 'train':
            self.model.train()
            ##################
            # forward pass
            ##################s
            rot, trans = inputs['rot'][0], inputs['trans'][0]
            src_pcd, tgt_pcd = inputs['src_points'].contiguous(), inputs['tgt_points'].contiguous()
            src_normals, tgt_normals = inputs['src_normals'].contiguous(), inputs['tgt_normals'].contiguous()
            src_feats, tgt_feats = inputs['src_feats'].contiguous(), inputs['tgt_feats'].contiguous()
            src_raw_pcd = inputs['raw_src_pcd'].contiguous()

            outputs = self.model.forward(src_pcd, tgt_pcd, src_feats, tgt_feats, src_normals, tgt_normals, rot, trans, src_raw_pcd)
            stats = self.loss_func(outputs, inputs)
            evaluator_stats = self.evaluator(outputs, inputs)
            stats.update(evaluator_stats)
            loss = stats['loss']
            if torch.isnan(stats['loss']):
                gt_src = torch.matmul(src_pcd + trans.T, rot.T)
                to_ply_file(gt_src, tgt_pcd)
                print(inputs['obj_id'], inputs['bbox'])
                stats['loss'] = torch.tensor(0., requires_grad=True)
                stats['c_loss'] = torch.tensor(0., requires_grad=True)
                stats['f_loss'] = torch.tensor(0., requires_grad=True)
                stats['o_loss'] = torch.tensor(0., requires_grad=True)
            loss.backward()

        else:
            self.model.eval()
            with torch.no_grad():
                rot, trans = inputs['rot'][0], inputs['trans'][0]
                src_pcd, tgt_pcd = inputs['src_points'].contiguous(), inputs['tgt_points'].contiguous()
                src_normals, tgt_normals = inputs['src_normals'].contiguous(), inputs['tgt_normals'].contiguous()
                src_feats, tgt_feats = inputs['src_feats'].contiguous(), inputs['tgt_feats'].contiguous()
                src_raw_pcd = inputs['raw_src_pcd'].contiguous()
                obj_id = inputs['obj_id'][0].item()
                og_id = inputs['og_id'][0].item()
                
                outputs = self.model.forward(src_pcd, tgt_pcd, src_feats, tgt_feats, src_normals, tgt_normals, rot, trans, src_raw_pcd, inputs['radius'][0])
                if outputs["gt_node_corr_overlaps"].shape[0] != 0:
                    stats = self.loss_func(outputs, inputs)
                    evaluator_stats = self.evaluator(outputs, inputs)

                    stats.update(evaluator_stats)
                
                
                # Save outputs
                sample_key = str(torch.mean(rot).item() + og_id)
                num_corrs = outputs["src_corr_points"].unique().shape[0]
                
                # Uncomment for GT correspondences
                
                # from scipy.spatial import cKDTree
                # tree = cKDTree(tgt_pcd.cpu().numpy())
                # gt_src = np.matmul(src_pcd.cpu().numpy() + trans.cpu().numpy().T, rot.cpu().numpy().T)
                # distances, indices = tree.query(gt_src, k=1)
                # close_points = distances < 0.07
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
                
        ########################################
        # re-organize dictionary stats
        ########################################
        stats['loss'] = float(stats['loss'].detach())
        stats['c_loss'] = float(stats['c_loss'].detach())
        stats['f_loss'] = float(stats['f_loss'].detach())
        stats['o_loss'] = float(stats['o_loss'].detach())
        return stats

    def inference_one_epoch(self, epoch, phase):
        '''
        Inference for an epoch
        :param epoch: current epoch
        :param phase: current phase of training
        :return:
        '''
        gc.collect()
        assert phase in ['train', 'val']

        #init stats meter
        stats_meter = self.stats_meter()

        num_iter = int(len(self.loader[phase]))
        c_loader_iter = self.loader[phase].__iter__()

        self.optimizer.zero_grad()
        idx = 0
        for c_iter in tqdm(range(num_iter)):
            inputs = next(c_loader_iter)
            for k, v in inputs.items():
                if type(v) == list:
                    inputs[k] = [item.to(self.device) for item in v]
                elif v is None:
                    inputs[k] = v
                else:
                    inputs[k] = v.to(self.device)

            if inputs["src_points"].shape[0] < 4096 or torch.sum(inputs["mask"]) < 4096:
                self.skipped+=1
                continue
            ######################
            # forward pass
            ######################
            
            stats = self.inference_one_batch(inputs, phase, idx=idx)
            idx += 1

            ######################
            # run optimization
            ######################
            if (c_iter + 1) % self.iter_size == 0 and phase == 'train':
                self.optimizer.step()
                self.optimizer.zero_grad()

            ########################
            # update to stats_meter
            ########################
            if inputs['obj_id'][0].item() == inputs['og_id'][0].item():
                for key, value in stats.items():
                    stats_meter[key].update(value)

            torch.cuda.empty_cache()

            if self.local_rank <= 0 and self.verbose and (c_iter + 1) % self.verbose_freq == 0:
                cur_iter = num_iter * (epoch - 1) + c_iter
                for key, value in stats_meter.items():
                    self.writer.add_scalar(f'{phase}/{key}', value.avg, cur_iter)

                message = f'{phase} Epoch: {epoch} [{c_iter + 1:4d}/{num_iter}] '
                for key, value in stats_meter.items():
                    message += f'{key}:{value.avg:.2f}\t'

                self.logger.write(message + '\n')
                
            if phase == 'train' and self.local_rank <= 0 and (c_iter + 1) % self.snapshot_interval == 0:
                if stats_meter['loss'].avg < self.best_train_loss: 
                    self.best_train_loss = stats_meter['loss'].avg
                    self._snapshot(epoch, 'best_train_loss')
                
        if self.local_rank <= 0:
            message = f'{phase} Epoch: {epoch} '
            for key, value in stats_meter.items():
                message += f'{key}: {value.avg:.4f}\t'

            self.logger.write(message + '\n')
        
        print("Skipped", self.skipped, "because of too small mask size")
        self.skipped = 0
        return stats_meter

    def train(self):
        '''
        Train
        :return:
        '''
        print('start training...')
        for epoch in range(self.start_epoch, self.max_epoch):
            if self.local_rank > -1:
                self.loader['train'].sampler.set_epoch(epoch)

            self.inference_one_epoch(epoch, 'train')
            self.scheduler.step()
            if epoch % self.config.val_every_n_epoch == 0:
                stats_meter = self.inference_one_epoch(epoch, 'val')
                if self.local_rank <= 0:
                    if stats_meter['loss'].avg < self.best_loss:
                        self.best_loss = stats_meter['loss'].avg
                        self._snapshot(epoch, 'best_loss')

                    if stats_meter['c_loss'].avg < self.best_c_loss:
                        self.best_c_loss = stats_meter['c_loss'].avg
                        self._snapshot(epoch, 'best_c_loss')

                    if stats_meter['f_loss'].avg < self.best_f_loss:
                        self.best_f_loss = stats_meter['f_loss'].avg
                        self._snapshot(epoch, 'best_f_loss')
                    if stats_meter['o_loss'].avg < self.best_o_loss:
                        self.best_o_loss = stats_meter['o_loss'].avg
                        self._snapshot(epoch, 'best_o_loss')

                    if stats_meter['PIR'].avg > self.best_PIR:
                        self.best_PIR = stats_meter['PIR'].avg
                        self._snapshot(epoch, 'best_PIR')

                    if stats_meter['IR'].avg > self.best_IR:
                        self.best_IR = stats_meter['IR'].avg
                        self._snapshot(epoch, 'best_IR')

        print('training finish!')

    def eval(self):
        '''
        Evaluation
        :return:
        '''
        print('start to evaluate on validation sets...')
        stats_meter = self.inference_one_epoch(0, 'val')
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
