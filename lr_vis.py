"""
ALL     https://www.cnblogs.com/wanghui-garcia/p/10895397.html
Cyclic  https://blog.csdn.net/zisuina_2/article/details/103236864
"""
import torch
from torch.optim.lr_scheduler import *
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.squeezenet import SqueezeNet


def get_lr(optimizer):
    # 只返回第1个 group 的 lr
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_scheduler(sc):
    if sc == 'multi':
        return MultiStepLR(optimizer, milestones=[int(num_epochs * i) for i in [0.5, 0.7, 0.9]], gamma=0.1)
    elif sc == 'step':
        return StepLR(optimizer, step_size=num_epochs // 5, gamma=0.1)

    elif sc == 'poly':  # 使用 lambda 实现 poly
        # lr = init_lr * lambda(e)  # 关于 epoch 的函数
        return LambdaLR(optimizer, lr_lambda=lambda e: (1 - e / num_epochs) ** 0.9)  # poly

    elif sc == 'cos':
        # lr = lr_min + 1 / 2 * (init_lr - lr_min) * (1 + cos(pi * T_cur / T_max)), [0, pi], [1, -1]
        return CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)  # cos 可设置 lr 最小值
    elif sc == 'cos_anneal':
        # T_max 半段 cos 函数跨越的 epochs 数量
        return CosineAnnealingLR(optimizer, T_max=num_epochs // 5, eta_min=1e-5)  # 5/2 个 cos 函数

    elif sc == 'expo':
        # lr = cur_lr * gamma = init_lr * gamma ** (e)
        return ExponentialLR(optimizer, gamma=0.9)  # expo 原始 lr 指数减少，gamma 越小，减少越快

    elif sc == 'cyclic':
        # 专注于每次 iteration
        return CyclicLR(optimizer, base_lr=init_lr, max_lr=init_lr * 2,
                        step_size_up=2, step_size_down=18,  # up=0 会 divide 0
                        cycle_momentum=True)  # 如果为 adam，用 False


def write_lr(optimizer, scheduler, tag):
    for epoch in range(1, num_epochs + 1):
        optimizer.step()
        scheduler.step()
        writer.add_scalar(tag, get_lr(optimizer), epoch)


if __name__ == '__main__':
    num_epochs = 100
    model = SqueezeNet()

    init_lr = 0.01
    writer = SummaryWriter(log_dir='runs/lr_vis')

    sches = ['multi', 'step', 'poly', 'cos', 'cos_anneal', 'expo']
    for sc in sches:
        print(sc)
        optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=0.9)
        scheduler = get_scheduler(sc)
        write_lr(optimizer, scheduler, tag=f'lr/{sc}')

    modes = ['triangular', 'triangular2', 'exp_range']
    for mode in modes:
        print(mode)
        optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=0.9)
        scheduler = CyclicLR(
            optimizer, base_lr=init_lr, max_lr=init_lr * 2,
            mode=mode,
            gamma=0.9 if mode == 'exp_range' else 1.,
            step_size_up=2, step_size_down=18,
        )
        write_lr(optimizer, scheduler, tag=f'CyclicLR/{mode}')
