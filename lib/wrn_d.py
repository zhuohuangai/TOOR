import torch
import torch.nn as nn
from easydl import *


class MyGradientReverseLayer(torch.autograd.Function):
    """
    usage:(can't be used in nn.Sequential, not a subclass of nn.Module)::

        x = Variable(torch.ones(1, 2), requires_grad=True)
        grl = GradientReverseLayer.apply
        y = grl(0.5, x)

        y.backward(torch.ones_like(y))

        print(x.grad)

    """
    @staticmethod
    def forward(ctx, coeff, input):
        ctx.coeff = coeff
        # this is necessary. if we just return ``input``, ``backward`` will not be called sometimes
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_outputs):
        coeff = ctx.coeff
        return None, -coeff * grad_outputs

class MyGradientReverseModule(nn.Module):
    """
    wrap GradientReverseLayer to be a nn.Module so that it can be used in ``nn.Sequential``

    usage::

        grl = GradientReverseModule(lambda step : aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=10000))

        x = Variable(torch.ones(1), requires_grad=True)
        ans = []
        for _ in range(10000):
            x.grad = None
            y = grl(x)
            y.backward()
            ans.append(variable_to_numpy(x.grad))

        plt.plot(list(range(10000)), ans)
        plt.show() # you can see gradient change from 0 to -1
    """
    def __init__(self, scheduler):
        super(MyGradientReverseModule, self).__init__()
        self.scheduler = scheduler
        self.register_buffer('global_step', torch.zeros(1))
        self.coeff = 0.0
        self.grl = MyGradientReverseLayer.apply

    def forward(self, x):
        if self.global_step.item() < 25000:
            self.coeff = 0
        else:
            self.coeff = self.scheduler(self.global_step.item() - 25000)
        if self.training:
            self.global_step += 1.0
        return self.grl(self.coeff, x)

def conv3x3(i_c, o_c, stride=1):
    return nn.Conv2d(i_c, o_c, 3, stride, 1, bias=False)

class BatchNorm2d(nn.BatchNorm2d):
    def __init__(self, channels, momentum=1e-3, eps=1e-3):
        super().__init__(channels)
        self.update_batch_stats = True

    def forward(self, x):
        if self.update_batch_stats:
            return super().forward(x)
        else:
            return nn.functional.batch_norm(
                x, None, None, self.weight, self.bias, True, self.momentum, self.eps
            )

def relu():
    return nn.LeakyReLU(0.1)

class residual(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1, activate_before_residual=False):
        super().__init__()
        layer = []
        if activate_before_residual:
            self.pre_act = nn.Sequential(
                BatchNorm2d(input_channels),
                relu()
            )
        else:
            self.pre_act = nn.Identity()
            layer.append(BatchNorm2d(input_channels))
            layer.append(relu())
        layer.append(conv3x3(input_channels, output_channels, stride))
        layer.append(BatchNorm2d(output_channels))
        layer.append(relu())
        layer.append(conv3x3(output_channels, output_channels))

        if stride >= 2 or input_channels != output_channels:
            self.identity = nn.Conv2d(input_channels, output_channels, 1, stride, bias=False)
        else:
            self.identity = nn.Identity()

        self.layer = nn.Sequential(*layer)

    def forward(self, x):
        x = self.pre_act(x)
        return self.identity(x) + self.layer(x)

class WRN(nn.Module):
    """ WRN28-width with leaky relu (negative slope is 0.1)"""
    def __init__(self, width, num_classes, transform_fn=None):
        super().__init__()

        self.init_conv = conv3x3(3, 16)

        filters = [16, 16*width, 32*width, 64*width]

        unit1 = [residual(filters[0], filters[1], activate_before_residual=True)] + \
            [residual(filters[1], filters[1]) for _ in range(1, 4)]
        self.unit1 = nn.Sequential(*unit1)

        unit2 = [residual(filters[1], filters[2], 2)] + \
            [residual(filters[2], filters[2]) for _ in range(1, 4)]
        self.unit2 = nn.Sequential(*unit2)

        unit3 = [residual(filters[2], filters[3], 2)] + \
            [residual(filters[3], filters[3]) for _ in range(1, 4)]
        self.unit3 = nn.Sequential(*unit3)

        self.unit4 = nn.Sequential(*[BatchNorm2d(filters[3]), relu(), nn.AdaptiveAvgPool2d(1)])

        # self.output = nn.Linear(filters[3], num_classes)
        self.cls = nn.Sequential(
            nn.Linear(filters[3], num_classes),
            nn.Softmax(dim=-1)
        )

        self.dis = nn.Sequential(
            nn.Linear(filters[3], 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024,1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        
        self.grl = MyGradientReverseModule(lambda step: aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=10000))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")    
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

        self.transform_fn = transform_fn
        self.__in_features = filters[3]

    def forward(self, x, return_feature=True):
        if self.training and self.transform_fn is not None:
            x = self.transform_fn(x)
        x = self.init_conv(x)
        x = self.unit1(x)
        x = self.unit2(x)
        x = self.unit3(x)
        f = self.unit4(x)
        f = f.squeeze()
        c = self.cls(f)
        reversed_f = self.grl(f)
        d = self.dis(reversed_f)

        return d, c

    def update_batch_stats(self, flag):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.update_batch_stats = flag


# class adversarialnet(nn.Module):
#     """
#     Discriminator network.
#     """
#     def __init__(self, in_feature):
#         super(adversarialnet, self).__init__()
#         self.main = nn.Sequential(
#             nn.Linear(in_feature, 1024),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(1024,1024),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(1024, 1),
#             nn.Sigmoid()
#         )
#         self.grl = GradientReverseModule(lambda step: aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=100000))

#     def forward(self, x):
#         x_ = self.grl(x)
#         y = self.main(x_)
#         return y