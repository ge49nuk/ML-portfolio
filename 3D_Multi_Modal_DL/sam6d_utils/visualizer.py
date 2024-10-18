import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from sam6d_provider.base_bop import OBJ_IDS


def vis_masks(img, masks, obj_ids, dataset_name, image_name='mask_visualization.png'):
    visualization = img
    pos_texts = []

    for i,mask in enumerate(masks):
        overlay = visualization.copy()
        np.random.seed(obj_ids[i])
        mask_color = np.random.randint(0, 256, size=(3,), dtype=np.uint8)
        np.random.seed(obj_ids[i])
        overlay[mask != 0] = mask_color
        alpha = 0.7
        visualization = cv2.addWeighted(overlay, alpha, visualization, 1 - alpha, 0)
        mask_indices = np.argwhere(mask != 0)
        centroid = np.mean(mask_indices, axis=0).astype(int)
        pos_texts.append((centroid, OBJ_IDS[dataset_name][obj_ids[i]]))
        
    plt.figure(figsize=(10, 10))
    plt.imshow(visualization)
    plt.axis('off')
    
    for pos_text in pos_texts:
        text = plt.text(pos_text[0][1], pos_text[0][0], f'obj_{pos_text[1]}', color='white', fontsize=12, 
        ha='center', va='center', fontweight='bold')
        text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='black')])
    
    plt.savefig(image_name, bbox_inches='tight', pad_inches=0)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.savefig('src_image.png', bbox_inches='tight', pad_inches=0)