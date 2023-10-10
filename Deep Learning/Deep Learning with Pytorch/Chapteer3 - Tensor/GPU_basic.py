import torch

points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]], dtype=torch.float)
points_gpu = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]], dtype=torch.float, device='cuda')
# 或者 to 到 GPU 上
points_gpu2 = points.to(device='cuda')
