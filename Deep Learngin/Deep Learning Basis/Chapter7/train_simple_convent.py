import numpy as np
from Convolution import *
import sys,os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from common.trainer import Trainer

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

max_epochs=20
network=SimpleConvNet(input_dim=(1,28,28),
                      conv_param={'filter_num':30,'filter_size':5,'pad':0,'stride':1},
                      hidden_size=100,output_size=10,weight_init_std=0.01)

# 通用 trainer
trainer=Trainer(network,x_train,t_train,x_test,t_test,epochs=max_epochs,mini_batch_size=100,
                optimizer='Adam',optimizer_param={'lr':0.001},
                evaluate_sample_num_per_epoch=1000)
trainer.train()

# 保存参数
network.save_params("params.pkl")
print("Saved Network Parameters!")