import os
import torch
import numpy as np
from tqdm import tqdm
from minsu3d.evaluation.semantic_segmentation import *
from minsu3d.evaluation.instance_segmentation import rle_decode, rle_encode


def save_prediction(save_path, all_pred_insts, mapping_ids, ignored_classes_indices):
    inst_pred_path = os.path.join(save_path, "instance")
    # SOFTGROUP
    inst_pred_masks_path = os.path.join(inst_pred_path, "predicted_masks")
    os.makedirs(inst_pred_masks_path, exist_ok=True)
    scan_instance_count = {}
    filtered_mapping_ids = [elem for i, elem in enumerate(mapping_ids) if i + 1 not in ignored_classes_indices]
    id_mappings = {}
    for i, label in enumerate(filtered_mapping_ids):
        id_mappings[i] = label
    for preds in tqdm(all_pred_insts, desc="==> Saving predictions ..."):
        tmp_info = []
        scan_id = preds[0]["scan_id"]
        for pred in preds:
            if scan_id not in scan_instance_count:
                scan_instance_count[scan_id] = 0
            mapped_label_id = id_mappings[pred['label_id'] - 1]
            tmp_info.append(
                f"predicted_masks/{scan_id}_{scan_instance_count[scan_id]:03d}.txt {mapped_label_id} {pred['conf']:.4f}")

            np.savetxt(
                os.path.join(inst_pred_masks_path, f"{scan_id}_{scan_instance_count[scan_id]:03d}.txt"),
                rle_decode(pred["pred_mask"]), fmt="%d")

            scan_instance_count[scan_id] += 1
        with open(os.path.join(inst_pred_path, f"{scan_id}.txt"), "w") as f:
            f.write("\n".join(tmp_info))
    

def save_prediction_joint_arch(save_path, predicted_verts, gt_verts, scan_desc_ids, descriptions, bboxes_pred, bboxes_gt):
    inst_pred_path = os.path.join(save_path, "instance")
    num_descriptions = len(scan_desc_ids)
    # JOINT_ARCH
    scenes_str = ""
    descriptions_str = ""
    for i in range(num_descriptions):
        num_queried = len(predicted_verts[i])
        id = scan_desc_ids[i]
        pred_verts_txt = ""
        gt_verts_txt = ""
        bboxes_pred_txt = ""
        bboxes_gt_txt = ""
        # TODO: save bboxes in good format
        for j in range(num_queried): # number of queried objects
            for vert in predicted_verts[i][j]: # number of verts
                pred_verts_txt += str(vert)+" "
            minxyz, maxxyz = bboxes_pred[i][j]
            minmaxxyz =  minxyz + maxxyz
            for val in minmaxxyz:
                bboxes_pred_txt += ("{:.4f}".format(val)) + " "
            if j < num_queried - 1:
                bboxes_pred_txt += ","
        for j in range(num_queried):
            for vert in gt_verts[i][j]:
                gt_verts_txt += str(vert)+" "
            minxyz, maxxyz = bboxes_gt[i][j]
            minmaxxyz =  minxyz + maxxyz
            for val in minmaxxyz:
                bboxes_gt_txt += ("{:.4f}".format(val)) + " "
            if j < num_queried - 1:
                bboxes_gt_txt += ","
        


        with open(os.path.join(inst_pred_path, f"{id}.txt"), "w") as f:
                f.write(pred_verts_txt +"\n"+ gt_verts_txt +"\n"+ bboxes_pred_txt + "\n" + bboxes_gt_txt)

        scenes_str += id + "\n"
        descriptions_str += id + " " + descriptions[i]+"\n"

    with open(os.path.join(inst_pred_path, "scenes.txt"), "w") as f:
                f.write(scenes_str)
    with open(os.path.join(inst_pred_path, "descriptions.txt"), "w") as f:
            f.write(descriptions_str)




def read_gt_files_from_disk(data_path):
    pth_file = torch.load(data_path)
    pth_file["xyz"] -= pth_file["xyz"].mean(axis=0)
    return pth_file["xyz"], pth_file["sem_labels"], pth_file["instance_ids"]


def read_pred_files_from_disk(data_path, gt_xyz, mapping_ids, ignored_classes_indices):

    sem_label_mapping = {}

    filtered_mapping_ids = [elem for i, elem in enumerate(mapping_ids) if i + 1 not in ignored_classes_indices]

    for i, item in enumerate(filtered_mapping_ids, 1):
        sem_label_mapping[item] = i
    pred_instances = []

    with open(data_path, "r") as f:
        for line in f:
            mask_relative_path, sem_label, confidence = line.strip().split()
            mask_path = os.path.join(os.path.dirname(data_path), mask_relative_path)
            pred_mask = np.loadtxt(mask_path, dtype=bool)
            pred = {"scan_id": os.path.basename(data_path), "label_id": sem_label_mapping[int(sem_label)],
                    "conf": float(confidence), "pred_mask": rle_encode(pred_mask)}
            pred_xyz = gt_xyz[pred_mask]
            pred["pred_bbox"] = np.concatenate((pred_xyz.min(0), pred_xyz.max(0)))
            pred_instances.append(pred)
    return pred_instances
