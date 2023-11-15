import tensorboardX
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import torch
from models import Discriminator, Generator
from utils import *
from datasets import ImageDataset
import itertools

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 超参数
    batchsize = 1
    size = 256  # 图片尺寸
    lr = 0.0002
    n_epoch = 200
    epoch = 0
    decay_epoch = 100  # 学习率衰减参数

    # networks
    # 两个生成器
    netG_A2B = Generator()
    netG_B2A = Generator()
    # 两个判别器
    netD_A = Discriminator()
    netD_B = Discriminator()

    # netG_A2B.to(device)
    # netG_B2A.to(device)
    # netD_A.to(device)
    # netD_B.to(device)

    # loss
    loss_GAN = torch.nn.MSELoss()
    loss_cycle = torch.nn.L1Loss()
    # 判断相似程度
    loss_identity = torch.nn.L1Loss()

    # optimizer & LR
    opt_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                             lr=lr, betas=(0.5, 0.9999))
    opt_DA = torch.optim.Adam(netD_A.parameters(), lr=lr, betas=(0.5, 0.9999))
    opt_DB = torch.optim.Adam(netD_B.parameters(), lr=lr, betas=(0.5, 0.9999))

    # 自适应学习率
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        opt_G,
        lr_lambda=LambdaLR(n_epoch, epoch, decay_epoch).step
    )
    lr_scheduler_DA = torch.optim.lr_scheduler.LambdaLR(
        opt_DA,
        lr_lambda=LambdaLR(n_epoch, epoch, decay_epoch).step
    )
    lr_scheduler_DB = torch.optim.lr_scheduler.LambdaLR(
        opt_DB,
        lr_lambda=LambdaLR(n_epoch, epoch, decay_epoch).step
    )

    # training
    data_root = "./datasets/apple2orange"
    input_A = torch.ones([1, 3, size, size], dtype=torch.float)
    input_B = torch.ones([1, 3, size, size], dtype=torch.float)
    # input_A.to(device)
    # input_B.to(device)

    # 正确的标签
    label_real = torch.ones([1], requires_grad=False, dtype=torch.float)
    label_fake = torch.zeros([1], requires_grad=False, dtype=torch.float)
    # label_real.to(device)
    # label_fake.to(device)

    # 两个对抗buffer
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # 日志
    log_path = "./logs"
    writer_log = tensorboardX.SummaryWriter(log_path)

    # 对图片的预处理
    transforms_ = [
        # 一般先将图片进行尺寸放大，再做 crop 操作
        transforms.Resize(int(256 * 1.12), Image.BICUBIC),
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    # 用官方写好的数据加载器轮子（能帮我们打乱数据以及利用多核加速之类的功能）
    dataloader = DataLoader(ImageDataset(data_root, transforms_), batch_size=batchsize, shuffle=True, num_workers=4)
    step = 0

    for epoch in range(n_epoch):
        for i, batch in enumerate(dataloader):
            # print(i, batch)
            real_A = torch.tensor(input_A.copy_(batch['A']), dtype=torch.float)
            real_B = torch.tensor(input_B.copy_(batch['B']), dtype=torch.float)
            # real_A.to(device)
            # real_B.to(device)

            opt_G.zero_grad()
            # 利用生成器A 生成 B
            same_B = netG_A2B(real_B)
            loss_identity_B = loss_identity(same_B, real_B) * 5.0
            # 同样方法生成器B 生成 A
            same_A = netG_B2A(real_A)
            loss_identity_A = loss_identity(same_A, real_A) * 5.0

            # 利用 A 生成假的 B
            fake_B = netG_A2B(real_A)
            # 然后看看当前的判别器的鉴假水平，计算 loss
            pred_fake = netD_B(fake_B)
            loss_GAN_A2B = loss_GAN(pred_fake, label_real)
            # 利用 B 生成假的 A
            fake_A = netG_B2A(real_B)
            pred_fake = netD_A(fake_A)
            loss_GAN_B2A = loss_GAN(pred_fake, label_real)

            # cycle loss
            # 利用生成器B，给定之前生成的假的 B数据，重新生成回 A
            recovered_A = netG_B2A(fake_B)
            loss_cycle_ABA = loss_cycle(recovered_A, real_A) * 10.0

            recovered_B = netG_A2B(fake_A)
            loss_cycle_BAB = loss_cycle(recovered_B, real_B) * 10.0

            # 生成器整体的 loss
            loss_G = loss_identity_A + loss_identity_B + \
                     loss_GAN_A2B + loss_GAN_B2A + \
                     loss_cycle_ABA + loss_cycle_BAB

            # 对生成器反向传播
            loss_G.backward()
            opt_G.step()

            ###################
            # 判别器部分
            opt_DA.zero_grad()

            # 生成器部分（也就是要欺骗的部分）
            # 这里是 判别器 A
            pred_real = netD_A(real_A)
            loss_D_real = loss_GAN(pred_real, label_real)

            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = netD_A(fake_A.detach())
            loss_D_fake = loss_GAN(pred_real, label_fake)

            # 判别器的 total loss
            loss_D_A = (loss_D_real + loss_D_fake) * 0.5
            loss_D_A.backward()
            opt_DA.step()

            # 判别器 B
            opt_DB.zero_grad()
            pred_real = netD_B(real_B)
            loss_D_real = loss_GAN(pred_real, label_real)

            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = netD_B(fake_B.detach())
            loss_D_fake = loss_GAN(pred_real, label_fake)

            # 判别器的 total loss
            loss_D_B = (loss_D_real + loss_D_fake) * 0.5
            loss_D_B.backward()
            opt_DB.step()

            # 打印对应的 loss
            print("loss_G: {}, loss_G_identity: {}, loss_G_GAN: {}"
                  .format(loss_G, loss_identity_A + loss_identity_B, loss_GAN_A2B + loss_GAN_B2A))
            print("loss_G_cycle: {}, loss_D_A: {}, loss_D_B: {}"
                  .format(loss_cycle_ABA + loss_cycle_BAB, loss_D_A, loss_D_B))

            # 将 loss 存入 log 中
            writer_log.add_scalar("loss_G", loss_G, global_step=step + 1)
            writer_log.add_scalar("loss_G_identity", loss_identity_A + loss_identity_B, global_step=step + 1)
            writer_log.add_scalar("loss_G_GAN", loss_GAN_A2B + loss_GAN_B2A, global_step=step + 1)
            writer_log.add_scalar("loss_cycle", loss_cycle_ABA + loss_cycle_BAB, global_step=step + 1)
            writer_log.add_scalar("loss_D_A", loss_D_A, global_step=step + 1)
            writer_log.add_scalar("loss_D_B", loss_D_B, global_step=step + 1)

            step += 1

        # 训练完一个 epoch 后，更新学习率，以及保存模型
        lr_scheduler_DA.step()
        lr_scheduler_DB.step()
        lr_scheduler_G.step()

        # 两个生成器和两个判别器
        torch.save(netG_A2B.state_dict(), "./models/netG_A2B.pth")
        torch.save(netG_B2A.state_dict(), "./models/netG_B2A.pth")
        torch.save(netD_A.state_dict(), "./models/netD_A.pth")
        torch.save(netD_B.state_dict(), "./models/netD_B.pth")
