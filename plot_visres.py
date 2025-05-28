import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import os

# 自定义类别颜色（RGB）
LABEL_COLORS = {
    1: (0, 0, 255),       # aorta - 蓝
    2: (0, 255, 0),       # gallbladder - 绿
    3: (255, 0, 0),       # left kidney - 红
    4: (0, 255, 255),     # right kidney - 青
    5: (255, 0, 255),     # liver - 品红
    6: (255, 255, 0),     # pancreas - 黄
    7: (0, 128, 255),     # spleen - 淡蓝
    8: (192, 192, 192)    # stomach - 灰
}

def decode_segmap_with_alpha(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 4), dtype=np.uint8)  # RGBA
    for label, color in LABEL_COLORS.items():
        region = mask == label
        color_mask[region, :3] = color
        color_mask[region, 3] = 160  # 半透明（0~255）
    return color_mask

# 文件路径
pred_dir = './spikeTU_Synapse224/TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_224'
save_path = './vis_result/SpikeTUNet'
os.makedirs(save_path, exist_ok=True)

case_list = ['case0001','case0002','case0003','case0004',
             'case0008','case0022','case0025','case0029',
             'case0032','case0035','case0036','case0038']

for _case_idx in case_list:
    
    img_path = os.path.join(pred_dir, _case_idx+"_img.nii.gz")
    gt_path = os.path.join(pred_dir, _case_idx+"_pred.nii.gz")
    
    img = nib.load(img_path).get_fdata()
    gt = nib.load(gt_path).get_fdata().astype(np.uint8)
    
    max_slice_index = gt.shape[2]
    
    for slice_index in range(max_slice_index):
        img_slice = img[:, :, slice_index]
        gt_color_rgba = decode_segmap_with_alpha(gt[:, :, slice_index])
    
        plt.figure(figsize=(5, 5))
        plt.imshow(img_slice, cmap='gray', interpolation='none')
        plt.imshow(gt_color_rgba, interpolation='none')  # 只标签区域可见，背景透明
        plt.axis('off')
    
        filename = _case_idx + f"_pred_slice{slice_index:04d}.png"
        plt.savefig(os.path.join(save_path, filename), bbox_inches='tight', pad_inches=0)
        plt.close()
