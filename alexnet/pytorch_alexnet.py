import torchvision.models as models
from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy as np
import math
import time
from datetime import datetime

batch_size = 24
num_batches = 600

alexnet = models.alexnet()

num_steps_burn_in = 10

optimizer = torch.optim.SGD(alexnet.parameters(), lr=0.01)

def run_benchmark_cpu():
    criterion = nn.CrossEntropyLoss()

    total_duration = 0.0
    total_duration_squared = 0.0
    for i in range(num_batches + num_steps_burn_in):
        #print(i)
        start_time = time.time()

        inp = Variable(torch.randn(batch_size, 3, 227, 227))
        labels = Variable(torch.ones(batch_size).type(torch.LongTensor))

        optimizer.zero_grad()

        outp = alexnet(inp)

        loss = criterion(outp, labels)
        loss.backward()
        optimizer.step()

        duration = time.time() - start_time
        if i > num_steps_burn_in:
            if not i % 10:
                print('%s: step %d, duration = %.3f' %
                (datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration

    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn * mn
    sd = math.sqrt(vr)
    print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
        (datetime.now(), "Forward_backword", num_batches, mn, sd))

def run_benchmark_gpu():
    criterion = nn.CrossEntropyLoss().cuda()

    total_duration = 0.0
    total_duration_squared = 0.0
    for i in range(num_batches + num_steps_burn_in):
        #print(i)
        start_time = time.time()

        inp = Variable(torch.randn(batch_size, 3, 227, 227)).cuda()
        labels = Variable(torch.ones(batch_size).type(torch.LongTensor)).cuda()

        optimizer.zero_grad()

        outp = alexnet.cuda().forward(inp)

        loss = criterion(outp, labels)
        loss.backward()
        optimizer.step()

        duration = time.time() - start_time
        if i > num_steps_burn_in:
            if not i % 10:
                print('%s: step %d, duration = %.3f' %
                (datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration

    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn * mn
    sd = math.sqrt(vr)
    print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
        (datetime.now(), "Forward_backword", num_batches, mn, sd))

run_benchmark_cpu()
#run_benchmark_gpu()

#torch.onnx.export(alexnet, inp, 'alexnet.onnx')

