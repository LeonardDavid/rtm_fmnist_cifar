import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import numpy as np
import random

import custommac1d

class Quantize(Function):
    @staticmethod
    def forward(ctx, input, quantization):
        output = input.clone().detach()
        output = quantization.applyQuantization(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None

quantize = Quantize.apply


class ErrorModel(Function):
    @staticmethod
    def forward(ctx, input, index_offset, block_size=64, error_model=None):
        output = input.clone().detach()
        # print(index_offset)
        # index_offset_tensor = torch.tensor(index_offset).clone().detach()
        # print(index_offset_tensor)
        # index_offset_clone = index_offset_tensor.clone().detach()
        output = error_model.applyErrorModel(output, index_offset, block_size)
        # print(output)
        return output
    
    # def forward(ctx, input, error_model=None):
    #     output = input.clone().detach()
    #     output = error_model.applyErrorModel(output)
    #     return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None, None

apply_error_model = ErrorModel.apply

# add for compatibility to every apply_error_model parameters that do not use index_offset and block_size
index_offset_default = np.zeros([2,2])
block_size_default = 1.0

def check_quantization(quantize_train, quantize_eval, training):
    condition = ((quantize_train == True) and (training == True)) or ((quantize_eval == True) and (training == False)) or ((quantize_train == True) and (quantize_eval == True))

    if (condition == True):
        return True
    else:
        return False


class QuantizedActivation(nn.Module):
    def __init__(self, *args, **kwargs):
        self.name = "QuantizedActivation"
        self.layerNR = kwargs.pop('layerNr', None)
        self.quantization = kwargs.pop('quantization', None)
        self.error_model = kwargs.pop('error_model', None)
        self.quantize_train = kwargs.pop('quantize_train', True)
        self.quantize_eval = kwargs.pop('quantize_eval', True)
        self.training = None
        super(QuantizedActivation, self).__init__(*args, **kwargs)

    def forward(self, input):
        output = None
        check_q = check_quantization(self.quantize_train,
         self.quantize_eval, self.training)
        if (check_q == True):
            output = quantize(input, self.quantization)
        else:
            output = input
        if self.error_model is not None:
            output = apply_error_model(output, index_offset_default, block_size_default, self.error_model)
        return output


class QuantizedLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        self.name = "QuantizedLinear"
        self.layerNR = kwargs.pop('layerNr', None)
        self.quantization = kwargs.pop('quantization', None)
        self.error_model = kwargs.pop('error_model', None)
        self.quantize_train = kwargs.pop('quantize_train', True)
        self.quantize_eval = kwargs.pop('quantize_eval', True)
        self.snn_sim = kwargs.pop('snn_sim', None)
        self.training = None
        self.test_rtm = kwargs.pop('test_rtm', False)
        self.index_offset = kwargs.pop('index_offset', None)
        self.lost_vals_r = kwargs.pop('lost_vals_r', None)
        self.lost_vals_l = kwargs.pop('lost_vals_l', None)
        self.block_size = kwargs.pop('block_size', None)
        self.protectLayers = kwargs.pop('protectLayers', None)
        self.err_shifts = kwargs.pop('err_shifts', None)
        super(QuantizedLinear, self).__init__(*args, **kwargs)

    def forward(self, input):
        if self.bias is None:
            # print("yes Lin", self.layerNR)
            quantized_weight = None
            check_q = check_quantization(self.quantize_train,
             self.quantize_eval, self.training)
            if (check_q == True):
                quantized_weight = quantize(self.weight, self.quantization)
            else:
                quantized_weight = self.weight
            if self.error_model is not None:
                # probability: error_model.p
                # print(self.error_model.p)
                # quantized_weight
                # print(quantized_weight)
                # print(quantized_weight.shape[0])
                # print(quantized_weight.shape[1])

                # print(self.layerNR)
                # print(self.protectLayers)

                if self.test_rtm is not None and self.protectLayers[self.layerNR-1]==0:
                    # print("yes Lin", self.layerNR)
                    # print(self.block_size)
                    # print("")
                    # print(np.sum(self.index_offset))

                    # nr_elem=0
                    # print(quantized_weight.shape[0])
                    # print(quantized_weight.shape[1])
                    # for i in range(0, quantized_weight.shape[0]):
                    #     for j in range(0, quantized_weight.shape[1]):
                    #         nr_elem += 1
                    #         # print(quantized_weight[i][j])
                    #     # print("\n")
                    # print(nr_elem)

                    err_shift = 0   # number of error shifts
                    shift = 0       # number of shifts (used for reading)
                    for i in range(0, self.index_offset.shape[0]):      #
                        for j in range(0, self.index_offset.shape[1]):  #
                            # start at 1 because AP is on the first element at the beginning, no shift is needed for reading the first value
                            for k in range(1, self.block_size):         #
                                shift += 1
                                if(random.uniform(0.0, 1.0) < self.error_model.p):
                                    err_shift += 1
                                    # 50/50 random possibility of right or left err_shift
                                    if(random.choice([-1,1]) == 1):
                                        # right err_shift
                                        if (self.index_offset[i][j] < self.block_size/2): # +1
                                            self.index_offset[i][j] += 1
                                        # self.index_offset[i][j] += 1
                                        # if (self.index_offset[i][j] > self.block_size/2): # +1
                                        #     self.lost_vals_r[i][j] += 1
                                        #     quantized_weight[i][(j+1)*self.block_size - int(self.lost_vals_r[i][j])] = random.choice([-1,1])
                                        # if (self.lost_vals_l[i][j] > 0):
                                        #     self.lost_vals_l[i][j] -= 1
                                    else:
                                        # left err_shift
                                        if (self.index_offset[i][j] < self.block_size/2): # -1
                                            self.index_offset[i][j] -= 1
                                        # self.index_offset[i][j] -= 1
                                        # if(-self.index_offset[i][j] > self.block_size/2): # +1
                                        #     self.lost_vals_l[i][j] += 1
                                        #     quantized_weight[i][j*self.block_size + int(self.lost_vals_l[i][j]) - 1] = random.choice([-1,1])
                                        # if(self.lost_vals_r[i][j] > 0):
                                        #     self.lost_vals_r[i][j] -= 1

                    self.err_shifts[self.layerNR-1] += err_shift

                    # print("local err_shifts: " + str(err_shift) + "/" + str(shift))
                    # print(self.err_shifts)

                    # print(np.sum(self.index_offset))
                    # print(self.index_offset)

                    # print(np.sum(self.lost_vals_r))
                    # print(np.sum(self.lost_vals_l))
                                        
                quantized_weight = ErrorModel.apply(quantized_weight, self.index_offset, self.block_size, self.error_model)
            if self.snn_sim is not None:
                wm_row = quantized_weight.shape[0]
                wm_col = quantized_weight.shape[1]
                im_col = input.shape[0]
                weight_b = quantized_weight
                input_b = input
                output_b = torch.zeros(im_col, wm_row).cuda()
                custommac1d.custommac1d(input_b, weight_b, output_b)
                output_b = output_b.detach()
                # print("custommac1d")
                ## check correctness
                # correct = torch.eq(output_b, output)
                # correct = (~correct).sum().item()
                # # 0 if tensors match
                # print("correctness: ", correct)
                output = F.linear(input, quantized_weight)
                output.data.copy_(output_b.data)
            else:
                output = F.linear(input, quantized_weight)
            return output
        else:
            quantized_weight = None
            quantized_bias = None
            check_q = check_quantization(self.quantize_train,
             self.quantize_eval, self.training)
            if (check_q == True):
                quantized_weight = quantize(self.weight, self.quantization)
                quantized_bias = quantize(self.bias, self.quantization)
            else:
                quantized_weight = self.weight
                quantized_bias = self.bias
            if self.error_model is not None:
                quantized_weight = apply_error_model(quantized_weight, index_offset_default, block_size_default, self.error_model)
                quantized_bias = apply_error_model(quantized_bias, index_offset_default, block_size_default, self.error_model)
            return F.linear(input, quantized_weight, quantized_bias)


class QuantizedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        self.name = "QuantizedConv2d"
        self.layerNR = kwargs.pop('layerNr', None)
        self.quantization = kwargs.pop('quantization', None)
        self.error_model = kwargs.pop('error_model', None)
        self.quantize_train = kwargs.pop('quantize_train', True)
        self.quantize_eval = kwargs.pop('quantize_eval', True)
        self.training = None
        self.test_rtm = kwargs.pop('test_rtm', False)
        self.index_offset = kwargs.pop('index_offset', None)
        self.lost_vals_r = kwargs.pop('lost_vals_r', None)
        self.lost_vals_l = kwargs.pop('lost_vals_l', None)
        self.block_size = kwargs.pop('block_size', None)
        self.protectLayers = kwargs.pop('protectLayers', None)
        self.err_shifts = kwargs.pop('err_shifts', None)
        super(QuantizedConv2d, self).__init__(*args, **kwargs)

    def forward(self, input):
        if self.bias is None:
            # print("yes 2D ", self.layerNR)
            quantized_weight = None
            check_q = check_quantization(self.quantize_train,
             self.quantize_eval, self.training)
            if (check_q == True):
                quantized_weight = quantize(self.weight, self.quantization)
            else:
                quantized_weight = self.weight
                quantized_bias = self.bias
            if self.error_model is not None:
                # probability: error_model.p
                # print(self.error_model.p)
                # quantized_weight
                # print(quantized_weight)
                # print("\n")
                # nr_elem=0
                # if(self.layerNR==1):
                #     print(quantized_weight.shape)
                #     print(quantized_weight.shape[0])
                #     print(quantized_weight.shape[1])
                #     for i in range(0, quantized_weight.shape[0]):
                #         for j in range(0, quantized_weight.shape[1]):
                #             nr_elem += 1
                #             # print(quantized_weight[i][j])
                #         # print("\n")
                #     print(nr_elem)
                

                # print(self.layerNR)
                # print(self.protectLayers)

                if self.test_rtm is not None and self.protectLayers[self.layerNR-1]==0:
                    # print("yes 2D ", self.layerNR)
                    # print(self.block_size)
                    # print("")
                    # print(np.sum(self.index_offset))

                    # nr_elem=0
                    # print(quantized_weight.shape[0])
                    # print(quantized_weight.shape[1])
                    # for i in range(0, quantized_weight.shape[0]):
                    #     for j in range(0, quantized_weight.shape[1]):
                    #         nr_elem += 1
                    #         # print(quantized_weight[i][j])
                    #     # print("\n")
                    # print(nr_elem)

                    # print(self.index_offset.shape[0])
                    # print(self.index_offset.shape[1])

                    err_shift = 0   # number of error shifts
                    shift = 0       # number of shifts (used for reading)
                    for i in range(0, self.index_offset.shape[0]):      # 
                        for j in range(0, self.index_offset.shape[1]):  # 
                            # start at 1 because AP is on the first element at the beginning, no shift is needed for reading the first value
                            for k in range(1, self.block_size):         # 
                                shift += 1
                                if(random.uniform(0.0, 1.0) < self.error_model.p):
                                    err_shift += 1
                                    # 50/50 random possibility of right or left err_shift
                                    if(random.choice([-1,1]) == 1):
                                        # right err_shift
                                        if (self.index_offset[i][j] < self.block_size/2): # +1
                                            self.index_offset[i][j] += 1
                                        # self.index_offset[i][j] += 1
                                        # if (self.index_offset[i][j] > self.block_size/2): # +1
                                        #     self.lost_vals_r[i][j] += 1
                                        #     quantized_weight[i][(j+1)*self.block_size - int(self.lost_vals_r[i][j])] = random.choice([-1,1])
                                        # if (self.lost_vals_l[i][j] > 0):
                                        #     self.lost_vals_l[i][j] -= 1
                                    else:
                                        # left err_shift
                                        if (self.index_offset[i][j] < self.block_size/2): # -1
                                            self.index_offset[i][j] -= 1
                                        # self.index_offset[i][j] -= 1
                                        # if(-self.index_offset[i][j] > self.block_size/2): # +1
                                        #     self.lost_vals_l[i][j] += 1
                                        #     quantized_weight[i][j*self.block_size + int(self.lost_vals_l[i][j]) - 1] = random.choice([-1,1])
                                        # if(self.lost_vals_r[i][j] > 0):
                                        #     self.lost_vals_r[i][j] -= 1

                    self.err_shifts[self.layerNR-1] += err_shift

                    # print("total err_shifts: " + str(err_shift) + "/" + str(shift))
                        
                    # print(self.err_shifts)

                    # print(np.sum(self.index_offset))
                    # print(np.max(self.index_offset))

                    # print(np.sum(self.lost_vals_r))
                    # print(np.sum(self.lost_vals_l))

                quantized_weight = apply_error_model(quantized_weight, self.index_offset, self.block_size, self.error_model)

            output = F.conv2d(input, quantized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            return output
        else:
            quantized_weight = None
            quantized_bias = None
            check_q = check_quantization(self.quantize_train,
             self.quantize_eval, self.training)
            # check quantization case
            if (check_q == True):
                quantized_weight = quantize(self.weight, self.quantization)
                quantized_bias = quantize(self.bias, self.quantization)
            else:
                quantized_weight = self.weight
                quantized_bias = self.bias
            # check whether error model needs to be applied
            if self.error_model is not None:
                quantized_weight = apply_error_model(quantized_weight, index_offset_default, block_size_default, self.error_model)
                quantized_bias = apply_error_model(quantized_bias, index_offset_default, block_size_default, self.error_model)
            # compute regular 2d conv
            output = F.conv2d(input, quantized_weight, quantized_bias, self.stride,
                              self.padding, self.dilation, self.groups)
            return output
