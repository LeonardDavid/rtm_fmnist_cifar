from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import time
import json
import sys
import os
from datetime import datetime
sys.path.append("code/python/")

from memory_profiler import profile

from Utils import Scale, Clippy, set_layer_mode, parse_args, dump_exp_data, create_exp_folder, store_exp_data, Criterion, binary_hingeloss

from QuantizedNN import QuantizedLinear, QuantizedConv2d, QuantizedActivation

from Models import VGG3

import binarizePM1
import binarizePM1FI
import quantization

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    set_layer_mode(model, "train") # propagate informaton about training to all layers

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # loss = F.nll_loss(output, target)
        criterion = nn.CrossEntropyLoss(reduction="none")
        loss = criterion(output, target).mean()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader, pr=1):
    model.eval()
    set_layer_mode(model, "eval") # propagate informaton about eval to all layers

    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for data, target in test_loader:
        # with data, target in test_loader:
            data, target = data.to(device), target.to(device)
            #TODO? index_offsets = model.getIndexOffsets().to(device)
            print("+")
            # print(data)
            # print(target)
            # print(test_loader)
            output = model(data)
            print("-")
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))
    if pr is not None:
        print('\nAccuracy: {:.2f}%\n'.format(
            100. * correct / len(test_loader.dataset)))

    accuracy = 100. * (correct / len(test_loader.dataset))

    return accuracy


# def test_error(model, device, test_loader):
#     model.eval()
#     set_layer_mode(model, "eval") # propagate informaton about eval to all layers

#     perrors = [i/100 for i in range(10)]
#     # perrors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#     # perrors = [0.1, 0.2]
#     # perrors = [0.2]

#     # perrors = [i/100 for i in range(20)]
#     # perrors.append(0.25)
#     # perrors.append(0.5)
#     # perrors.append(0.75)
#     # perrors.append(0.9)
#     # perrors.append(1.0)

#     # perrors = [0.1, 0.25, 0.5, 0.75, 0.9]

#     print("error rates: ", perrors)

#     all_accuracies = []
#     accuracies = []
#     for perror in perrors:
        
#         print("start")
#         print(model.printIndexOffsets())
#         # print(model.printLostValsR())
#         # print(model.printLostValsL())

#         # update perror in every layer
#         for layer in model.children():
#             if isinstance(layer, (QuantizedActivation, QuantizedLinear, QuantizedConv2d)):
#                 if layer.error_model is not None:
#                     layer.error_model.updateErrorModel(perror)

#         print("Error rate: ", perror)
#         accuracy = test(model, device, test_loader)
#         all_accuracies.append(
#             {
#                 "perror":perror,
#                 "accuracy": accuracy
#             }
#         )
#         accuracies.append(accuracy)
        
#         print("end")
#         print(model.printIndexOffsets())
#         # print(model.printLostValsR())
#         # print(model.printLostValsL())

#         # model.resetOffsets()

#     # reset error models
#     for layer in model.children():
#         if isinstance(layer, (QuantizedActivation, QuantizedLinear, QuantizedConv2d)):
#             if layer.error_model is not None:
#                 layer.error_model.resetErrorModel()

#     print(perrors)
#     print(accuracies)

#     return all_accuracies

# @profile
def test_error(model, device, test_loader, perror):
    model.eval()
    set_layer_mode(model, "eval") # propagate informaton about eval to all layers
       
    # print("start")
    # print(model.printIndexOffsets())
    # print(model.printLostValsR())
    # print(model.printLostValsL())

    # update perror in every layer
    for layer in model.children():
        if isinstance(layer, (QuantizedActivation, QuantizedLinear, QuantizedConv2d)):
            if layer.error_model is not None:
                layer.error_model.updateErrorModel(perror)

    print("Error rate: ", perror)
    accuracy = test(model, device, test_loader)
    
    print("total_err_shifts: ", model.err_shifts)
    # print("end")
    # print(model.printIndexOffsets())
    # print(model.printLostValsR())
    # print(model.printLostValsL())

    # if model.getLostValsSum() != 0:
    #     print("Lost some values!")
    #     print(model.printIndexOffsets())
    #     print(model.printLostValsR())
    #     print(model.printLostValsL())

    # reset error models
    for layer in model.children():
        if isinstance(layer, (QuantizedActivation, QuantizedLinear, QuantizedConv2d)):
            if layer.error_model is not None:
                layer.error_model.resetErrorModel()

    # print(str(perror) + " " + str(accuracy))

    return accuracy
