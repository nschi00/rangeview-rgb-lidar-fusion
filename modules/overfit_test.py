import torch
import torch.nn as nn
from losses.Lovasz_Softmax import Lovasz_softmax
from losses.boundary_loss import BoundaryLoss
import time
from ioueval import *

def overfit_test(model, batchsize, rgb=False):
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    pytorch_total_params = sum(p.numel()
                               for p in model.parameters() if p.requires_grad)
    print("Number of parameters: ", pytorch_total_params / 1000000, "M")
    time_train = []
    
    evaluator = iouEval(20, device='cuda', ignore=0)

    input_3D = torch.randn(batchsize, 5, 64, 512).cuda()
    input_rgb = torch.rand(batchsize, 3, 64, 512).cuda()
    label = torch.randint(0, 20, (batchsize, 64, 512)).cuda()
    criterion = nn.NLLLoss().cuda()
    ls = Lovasz_softmax(ignore=0).cuda()
    bd = BoundaryLoss().cuda()
    for i in range(20000):
        
        model.train()
        #with torch.no_grad():
        start_time = time.time()
        if rgb == False:
            out = model(input_3D, label)
        else:
            out = model(input_3D, input_rgb)
        if type(out) != list:
            out = [out]
        for j in range(len(out)):
            if j == 0:
                bdlosss = bd(out[j], label.long())
                loss_mn = criterion(torch.log(out[j].clamp(
                    min=1e-8)), label) + 1.5 * ls(out[j], label.long())
            else:
                bdlosss += bd(out[j], label.long())
                loss_mn += criterion(torch.log(out[j].clamp(
                    min=1e-8)), label) + 1.5 * ls(out[j], label.long())

        loss_m = loss_mn + bdlosss
        output = out[0]
                
        optimizer.zero_grad()
        loss_m.backward()
        optimizer.step()
        
        evaluator.addBatch(output.argmax(dim=1), label)
        accuracy = evaluator.getacc()
        jaccard, _ = evaluator.getIoUMissingClass()
        
        fwt = time.time() - start_time
        time_train.append(fwt)
        
        print("Loss: ", loss_m.mean().item())
        print("Accuracy: ", accuracy.item())
        print("IoU: ", jaccard.item())
        print("Time:", fwt)