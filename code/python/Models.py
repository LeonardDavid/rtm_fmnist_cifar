import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from QuantizedNN import QuantizedLinear, QuantizedConv2d, QuantizedActivation

from Utils import Scale, Clippy

class FC(nn.Module):
    def __init__(self, quantMethod=None, snn_sim=None, quantize_train=True, quantize_eval=True, error_model=None):
        super(FC, self).__init__()
        self.name = "FC"
        self.quantization = quantMethod
        self.q_train = quantize_train
        self.q_test = quantize_eval
        self.error_model = error_model
        self.snn_sim = snn_sim
        self.htanh = nn.Hardtanh()

        self.flatten = torch.flatten
        self.fcfc1 = QuantizedLinear(28*28, 2048, quantization=self.quantization, snn_sim=self.snn_sim, error_model=self.error_model, layerNr=1, bias=False)
        self.fcbn1 = nn.BatchNorm1d(2048)
        self.fcqact1 = QuantizedActivation(quantization=self.quantization)

        self.fcfc2 = QuantizedLinear(2048, 2048, quantization=self.quantization, snn_sim=self.snn_sim, error_model=self.error_model, layerNr=2, bias=False)
        self.fcbn2 = nn.BatchNorm1d(2048)
        self.fcqact2 = QuantizedActivation(quantization=self.quantization)
        self.fcfc3 = QuantizedLinear(2048, 10, quantization=self.quantization, snn_sim=self.snn_sim, layerNr=3, bias=False)
        self.scale = Scale()

    def forward(self, x):
        x = self.flatten(x, start_dim=1, end_dim=3)
        x = self.fcfc1(x)
        x = self.fcbn1(x)
        x = self.htanh(x)
        x = self.fcqact1(x)

        x = self.fcfc2(x)
        x = self.fcbn2(x)
        x = self.htanh(x)
        x = self.fcqact2(x)

        x = self.fcfc3(x)
        x = self.scale(x)

        return x

class VGG3(nn.Module):
    def __init__(self, quantMethod=None, quantize_train=True, quantize_eval=True, error_model=None, test_rtm = None, block_size=64):
        super(VGG3, self).__init__()
        self.name = "VGG3"
        self.quantization = quantMethod
        self.q_train = quantize_train
        self.q_test = quantize_eval
        self.error_model = error_model
        self.htanh = nn.Hardtanh()
        self.block_size = block_size # 64
        self.resetOffsets()

        # index_offset_default = np.zeros([2,2])
        # lost_vals_l_default = np.zeros([2,2])
        # lost_vals_r_default = np.zeros([2,2])
        # block_size_default = 1.0

        # self.conv1_size = (1, 64)
        self.conv1 = QuantizedConv2d(1, 64, kernel_size=3, padding=1, stride=1, quantization=self.quantization, error_model=self.error_model, test_rtm = test_rtm, index_offset = self.index_offset_conv1, lost_vals_r = self.lost_vals_r_conv1, lost_vals_l = self.lost_vals_l_conv1, block_size = self.block_size, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.qact1 = QuantizedActivation(quantization=self.quantization)

        self.conv2 = QuantizedConv2d(64, 64, kernel_size=3, padding=1, stride=1, quantization=self.quantization, error_model=self.error_model, test_rtm = test_rtm, index_offset = self.index_offset_conv2, lost_vals_r = self.lost_vals_r_conv2, lost_vals_l = self.lost_vals_l_conv2, block_size = self.block_size, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.qact2 = QuantizedActivation(quantization=self.quantization)

        self.fc1 = QuantizedLinear(7*7*64, 2048, quantization=self.quantization, error_model=self.error_model, test_rtm = test_rtm, index_offset = self.index_offset_fc1, lost_vals_r = self.lost_vals_r_fc1, lost_vals_l = self.lost_vals_l_fc1, block_size = self.block_size, bias=False)
        self.bn3 = nn.BatchNorm1d(2048)
        self.qact3 = QuantizedActivation(quantization=self.quantization)

        self.fc2 = QuantizedLinear(2048, 10, quantization=self.quantization, error_model=self.error_model, test_rtm = test_rtm, index_offset = self.index_offset_fc2, lost_vals_r = self.lost_vals_r_fc2, lost_vals_l = self.lost_vals_l_fc2, block_size = self.block_size, bias=False)
        self.scale = Scale()

    def getBlockSize(self):
        return self.block_size
    
    def resetOffsets(self):
        # if self.conv1_size(0) >= 64:
        #     nr_blocks_conv1 = int(self.conv1_size(0)/self.block_size)
        # else:
        #     nr_blocks_conv1 = self.conv1_size
        # for conv 1 nr_blocks_conv1 has to be 1, because else it will set it to 0
        self.index_offset_conv1 = np.zeros((64, 1))
        self.lost_vals_r_conv1 = np.zeros((self.index_offset_conv1.shape[0], self.index_offset_conv1.shape[1]))
        self.lost_vals_l_conv1 = np.zeros((self.index_offset_conv1.shape[0], self.index_offset_conv1.shape[1]))

        self.index_offset_conv2 = np.zeros((64, int(64/self.block_size)))
        self.lost_vals_r_conv2 = np.zeros((self.index_offset_conv2.shape[0], self.index_offset_conv2.shape[1]))
        self.lost_vals_l_conv2 = np.zeros((self.index_offset_conv2.shape[0], self.index_offset_conv2.shape[1]))

        self.index_offset_fc1 = np.zeros((2048, int(7*7*64/self.block_size)))
        self.lost_vals_r_fc1 = np.zeros((self.index_offset_fc1.shape[0], self.index_offset_fc1.shape[1]))
        self.lost_vals_l_fc1 = np.zeros((self.index_offset_fc1.shape[0], self.index_offset_fc1.shape[1]))

        self.index_offset_fc2 = np.zeros((10, int(2048/self.block_size)))
        self.lost_vals_r_fc2 = np.zeros((self.index_offset_fc2.shape[0], self.index_offset_fc2.shape[1]))
        self.lost_vals_l_fc2 = np.zeros((self.index_offset_fc2.shape[0], self.index_offset_fc2.shape[1]))

    
    def getLostValsSum(self):
        return np.sum(self.lost_vals_l_conv1) + np.sum(self.lost_vals_l_conv2) + np.sum(self.lost_vals_l_fc1) + np.sum(self.lost_vals_l_fc2) + np.sum(self.lost_vals_r_fc1) + np.sum(self.lost_vals_r_fc2)
    
    def printIndexOffsets(self):
        print("conv1 " + str(self.index_offset_conv1.shape[0]) + " " + str(self.index_offset_conv1.shape[1]) + " " + str(np.sum(self.index_offset_conv1)))
        print(self.index_offset_conv1)
        print("conv2 " + str(self.index_offset_conv2.shape[0]) + " " + str(self.index_offset_conv2.shape[1]) + " " + str(np.sum(self.index_offset_conv2)))
        print(self.index_offset_conv2)
        # print("fc1 " + str(self.index_offset_fc1.shape[0]) + " " + str(self.index_offset_fc1.shape[1]) + " " + str(np.sum(self.index_offset_fc1)))
        # print(self.index_offset_fc1)
        # print("fc2 " + str(self.index_offset_fc2.shape[0]) + " " + str(self.index_offset_fc2.shape[1]) + " " + str(np.sum(self.index_offset_fc2)))
        # print(self.index_offset_fc2)

    def printLostValsR(self):
        print("lvr_conv1 " + str(self.lost_vals_r_conv1.shape[0]) + " " + str(self.lost_vals_r_conv1.shape[1]) + " " + str(np.sum(self.lost_vals_r_conv1)))
        print(self.lost_vals_r_conv1)
        print("lvr_conv2 " + str(self.lost_vals_r_conv2.shape[0]) + " " + str(self.lost_vals_r_conv2.shape[1]) + " " + str(np.sum(self.lost_vals_r_conv2)))
        print(self.lost_vals_r_conv2)
        # print("lvr_fc1 " + str(self.lost_vals_r_fc1.shape[0]) + " " + str(self.lost_vals_r_fc1.shape[1]) + " " + str(np.sum(self.lost_vals_r_fc1)))
        # print(self.lost_vals_r_fc1)
        # print("lvr_fc2 " + str(self.lost_vals_r_fc2.shape[0]) + " " + str(self.lost_vals_r_fc2.shape[1]) + " " + str(np.sum(self.lost_vals_r_fc2)))
        # print(self.lost_vals_r_fc2)

    def printLostValsL(self):
        print("lvl_conv1 " + str(self.lost_vals_l_conv1.shape[0]) + " " + str(self.lost_vals_l_conv1.shape[1]) + " " + str(np.sum(self.lost_vals_l_conv1)))
        print(self.lost_vals_l_conv1)
        print("lvl_conv2 " + str(self.lost_vals_l_conv2.shape[0]) + " " + str(self.lost_vals_l_conv2.shape[1]) + " " + str(np.sum(self.lost_vals_l_conv2)))
        print(self.lost_vals_l_conv2)
        # print("lvl_fc1 " + str(self.lost_vals_l_fc1.shape[0]) + " " + str(self.lost_vals_l_fc1.shape[1]) + " " + str(np.sum(self.lost_vals_l_fc1)))
        # print(self.lost_vals_l_fc1)
        # print("lvl_fc2 " + str(self.lost_vals_l_fc2.shape[0]) + " " + str(self.lost_vals_l_fc2.shape[1]) + " " + str(np.sum(self.lost_vals_l_fc2)))
        # print(self.lost_vals_l_fc2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.bn1(x)
        x = self.htanh(x)
        x = self.qact1(x)

        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.bn2(x)
        x = self.htanh(x)
        x = self.qact2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.htanh(x)
        x = self.qact2(x)

        x = self.fc2(x)
        x = self.scale(x)

        return x


class VGG7(nn.Module):
    def __init__(self, quantMethod=None, quantize_train=True, quantize_eval=True, error_model=None, test_rtm = None, block_size=64):
        super(VGG7, self).__init__()
        self.name = "VGG7"
        self.quantization = quantMethod
        self.q_train = quantize_train
        self.q_test = quantize_eval
        self.error_model = error_model
        self.block_size = block_size
        self.htanh = nn.Hardtanh()
        self.resetOffsets()

        #CNN
        # block 1
        self.conv1 = QuantizedConv2d(3, 128, kernel_size=3, padding=1, stride=1, quantization=self.quantization, layerNr=1, error_model=self.error_model, test_rtm = test_rtm, index_offset = self.index_offset_conv1, lost_vals_r = self.lost_vals_r_conv1, lost_vals_l = self.lost_vals_l_conv1, block_size = self.block_size, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.qact1 = QuantizedActivation(quantization=self.quantization)

        # block 2
        self.conv2 = QuantizedConv2d(128, 128, kernel_size=3, padding=1, stride=1, quantization=self.quantization, layerNr=2, error_model=self.error_model, test_rtm = test_rtm, index_offset = self.index_offset_conv2, lost_vals_r = self.lost_vals_r_conv2, lost_vals_l = self.lost_vals_l_conv2, block_size = self.block_size, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.qact2 = QuantizedActivation(quantization=self.quantization)

        # block 3
        self.conv3 = QuantizedConv2d(128, 256, kernel_size=3, padding=1, stride=1, quantization=self.quantization, layerNr=3, error_model=self.error_model, test_rtm = test_rtm, index_offset = self.index_offset_conv3, lost_vals_r = self.lost_vals_r_conv3, lost_vals_l = self.lost_vals_l_conv3, block_size = self.block_size, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.qact3 = QuantizedActivation(quantization=self.quantization)

        # block 4
        self.conv4 = QuantizedConv2d(256, 256, kernel_size=3, padding=1, stride=1, quantization=self.quantization, layerNr=4, error_model=self.error_model, test_rtm = test_rtm, index_offset = self.index_offset_conv4, lost_vals_r = self.lost_vals_r_conv4, lost_vals_l = self.lost_vals_l_conv4, block_size = self.block_size, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        self.qact4 = QuantizedActivation(quantization=self.quantization)

        # block 5
        self.conv5 = QuantizedConv2d(256, 512, kernel_size=3, padding=1, stride=1, quantization=self.quantization, layerNr=5, error_model=self.error_model, test_rtm = test_rtm, index_offset = self.index_offset_conv5, lost_vals_r = self.lost_vals_r_conv5, lost_vals_l = self.lost_vals_l_conv5, block_size = self.block_size, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        self.qact5 = QuantizedActivation(quantization=self.quantization)

        # block 6
        self.conv6 = QuantizedConv2d(512, 512, kernel_size=3, padding=1, stride=1, quantization=self.quantization, layerNr=6, error_model=self.error_model, test_rtm = test_rtm, index_offset = self.index_offset_conv6, lost_vals_r = self.lost_vals_r_conv6, lost_vals_l = self.lost_vals_l_conv6, block_size = self.block_size, bias=False)
        self.bn6 = nn.BatchNorm2d(512)
        self.qact6 = QuantizedActivation(quantization=self.quantization)

        # block 7
        self.fc1 = QuantizedLinear(8192, 1024, quantization=self.quantization, layerNr=7, error_model=self.error_model, test_rtm = test_rtm, index_offset = self.index_offset_fc1, lost_vals_r = self.lost_vals_r_fc1, lost_vals_l = self.lost_vals_l_fc1, block_size = self.block_size, bias=False)
        self.bn7 = nn.BatchNorm1d(1024)
        self.qact7 = QuantizedActivation(quantization=self.quantization)

        self.fc2 = QuantizedLinear(1024, 10, quantization=self.quantization, layerNr=8, error_model=self.error_model, test_rtm = test_rtm, index_offset = self.index_offset_fc2, lost_vals_r = self.lost_vals_r_fc2, lost_vals_l = self.lost_vals_l_fc2, block_size = self.block_size, bias=False)
        self.scale = Scale(init_value=1e-3)

    
    def getBlockSize(self):
        return self.block_size
    
    
    def resetOffsets(self):
        # if self.conv1_size(0) >= 64:
        #     nr_blocks_conv1 = int(self.conv1_size(0)/self.block_size)
        # else:
        #     nr_blocks_conv1 = self.conv1_size
        # for conv 1 nr_blocks_conv1 has to be 3, because else it will set it to 0
        self.index_offset_conv1 = np.zeros((128, 3))
        self.lost_vals_r_conv1 = np.zeros((self.index_offset_conv1.shape[0], self.index_offset_conv1.shape[1]))
        self.lost_vals_l_conv1 = np.zeros((self.index_offset_conv1.shape[0], self.index_offset_conv1.shape[1]))

        self.index_offset_conv2 = np.zeros((128, int(128/self.block_size)))
        self.lost_vals_r_conv2 = np.zeros((self.index_offset_conv2.shape[0], self.index_offset_conv2.shape[1]))
        self.lost_vals_l_conv2 = np.zeros((self.index_offset_conv2.shape[0], self.index_offset_conv2.shape[1]))

        self.index_offset_conv3 = np.zeros((256, int(128/self.block_size)))
        self.lost_vals_r_conv3 = np.zeros((self.index_offset_conv3.shape[0], self.index_offset_conv3.shape[1]))
        self.lost_vals_l_conv3 = np.zeros((self.index_offset_conv3.shape[0], self.index_offset_conv3.shape[1]))

        self.index_offset_conv4 = np.zeros((256, int(256/self.block_size)))
        self.lost_vals_r_conv4 = np.zeros((self.index_offset_conv4.shape[0], self.index_offset_conv4.shape[1]))
        self.lost_vals_l_conv4 = np.zeros((self.index_offset_conv4.shape[0], self.index_offset_conv4.shape[1]))

        self.index_offset_conv5 = np.zeros((512, int(256/self.block_size)))
        self.lost_vals_r_conv5 = np.zeros((self.index_offset_conv5.shape[0], self.index_offset_conv5.shape[1]))
        self.lost_vals_l_conv5 = np.zeros((self.index_offset_conv5.shape[0], self.index_offset_conv5.shape[1]))

        self.index_offset_conv6 = np.zeros((512, int(512/self.block_size)))
        self.lost_vals_r_conv6 = np.zeros((self.index_offset_conv6.shape[0], self.index_offset_conv6.shape[1]))
        self.lost_vals_l_conv6 = np.zeros((self.index_offset_conv6.shape[0], self.index_offset_conv6.shape[1]))

        self.index_offset_fc1 = np.zeros((1024, int(8192/self.block_size)))
        self.lost_vals_r_fc1 = np.zeros((self.index_offset_fc1.shape[0], self.index_offset_fc1.shape[1]))
        self.lost_vals_l_fc1 = np.zeros((self.index_offset_fc1.shape[0], self.index_offset_fc1.shape[1]))

        self.index_offset_fc2 = np.zeros((10, int(1024/self.block_size)))
        self.lost_vals_r_fc2 = np.zeros((self.index_offset_fc2.shape[0], self.index_offset_fc2.shape[1]))
        self.lost_vals_l_fc2 = np.zeros((self.index_offset_fc2.shape[0], self.index_offset_fc2.shape[1]))


    def forward(self, x):

        # block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.htanh(x)
        x = self.qact1(x)

        # block 2
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.bn2(x)
        x = self.htanh(x)
        x = self.qact2(x)

        # block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.htanh(x)
        x = self.qact3(x)

        # block 4
        x = self.conv4(x)
        x = F.max_pool2d(x, 2)
        x = self.bn4(x)
        x = self.htanh(x)
        x = self.qact4(x)

        # block 5
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.htanh(x)
        x = self.qact5(x)

        # block 6
        x = self.conv6(x)
        x = F.max_pool2d(x, 2)
        x = self.bn6(x)
        x = self.htanh(x)
        x = self.qact6(x)

        # block 7
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn7(x)
        x = self.htanh(x)
        x = self.qact3(x)

        x = self.fc2(x)
        x = self.scale(x)

        return x

class BNN_FASHION_FC(nn.Module):
    def __init__(self, quantMethod):
        super(BNN_FASHION_FC, self).__init__()
        self.quantization = quantMethod

        self.htanh = nn.Hardtanh()
        self.flatten = torch.flatten
        self.fcfc1 = QuantizedLinear(28*28, 2048, quantization=self.quantization, layerNr=1)
        self.fcbn1 = nn.BatchNorm1d(2048)
        self.fcqact1 = QuantizedActivation(quantization=self.quantization)

        self.fcfc2 = QuantizedLinear(2048, 2048, quantization=self.quantization, layerNr=2)
        self.fcbn2 = nn.BatchNorm1d(2048)
        self.fcqact2 = QuantizedActivation(quantization=self.quantization)
        self.fcfc3 = QuantizedLinear(2048, 10, quantization=self.quantization, layerNr=3)
        self.scale = Scale(init_value=1e-3)

    def forward(self, x):
        x = self.flatten(x, start_dim=1, end_dim=3)
        x = self.fcfc1(x)
        x = self.fcbn1(x)
        x = self.htanh(x)
        x = self.fcqact1(x)

        x = self.fcfc2(x)
        x = self.fcbn2(x)
        x = self.htanh(x)
        x = self.fcqact2(x)

        x = self.fcfc3(x)
        x = self.scale(x)

        return x
