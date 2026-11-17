import torch.nn as nn
import numpy

class BinOp():
    def __init__(self, model, param):
        # count the number of Conv2d and Linear
        count_targets = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                count_targets = count_targets + 1

        if param == 0:
            # non layer binarize
            start_range = 1
            end_range = count_targets-2
        elif param == 1:
            # first layer binarize
            start_range = 0
            end_range = count_targets-2
        elif param == 2:
            # last layer binarize
            start_range = 1
            end_range = count_targets-1
        elif param == 3:
            # full layer binarize
            start_range = 0
            end_range = count_targets-1

        self.bin_range = numpy.linspace(start_range, end_range, end_range-start_range+1).astype('int').tolist()
        self.num_of_params = len(self.bin_range)
        self.saved_params = []
        self.target_modules = []
        index = -1
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                index = index + 1
                if index in self.bin_range:
                    tmp = m.weight.data.clone()
                    self.saved_params.append(tmp)
                    self.target_modules.append(m.weight)
        return

    def binarization(self):
        self.meancenterConvParams()
        self.clampConvParams()
        self.save_params()
        self.ternarizeConvParams()

    def meancenterConvParams(self):
        for index in range(self.num_of_params):
            s = self.target_modules[index].data.size()
            negMean = self.target_modules[index].data.mean(1, keepdim=True).mul(-1).expand_as(self.target_modules[index].data)
            self.target_modules[index].data = self.target_modules[index].data.add(negMean)

    def clampConvParams(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data = self.target_modules[index].data.clamp(-1.0, 1.0)

    def save_params(self):
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def binarizeConvParams(self):
        for index in range(self.num_of_params):
            n = self.target_modules[index].data[0].nelement()
            s = self.target_modules[index].data.size()

            if len(s) == 4: #if convolutional layer
                m = self.target_modules[index].data.norm(1, 3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True).div(n)
            elif len(s) == 2: #if fully connected layer
                m = self.target_modules[index].data.norm(1, 1, keepdim=True).div(n)

            self.target_modules[index].data = self.target_modules[index].data.sign().mul(m.expand(s))

    def ternarizeConvParams(self):
        tFactor = 0.07
        for index in range(self.num_of_params):
            input = self.target_modules[index].data
            max_w = input.abs().max()
            th = tFactor * max_w
            output = input.clone().zero_()

            W = input[input.ge(th) + input.le(-th)].abs().mean()
            output[input.ge(th)] = W
            output[input.lt(-th)] = -W

            self.target_modules[index].data = output

    def restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])
