import torch
import torch.nn as nn
import torch.optim as optim

linear_model = nn.Linear(1, 1)
print(linear_model)
print(linear_model.weight)
print(linear_model.bias)

x = torch.ones(1)
print(linear_model(x))

x2 = torch.ones(10, 1)
print("batched data: ")
print(linear_model(x2))

t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c).unsqueeze(1)
t_u = torch.tensor(t_u).unsqueeze(1)

optimizer=optim.SGD(
    linear_model.parameters(),
    lr=1e-2
)