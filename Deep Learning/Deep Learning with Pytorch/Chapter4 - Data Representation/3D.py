import imageio.v2 as imageio
import torch

dir_path = "../data/p1ch4/volumetric-dicom/2-LUNG 3.0  B70f-04083"
vol_arr = imageio.volread(dir_path, 'DICOM')
print(vol_arr.shape)

# 加个 通道维度
vol = torch.from_numpy(vol_arr)
vol = torch.unsqueeze(vol, 0)
print(vol.shape)
