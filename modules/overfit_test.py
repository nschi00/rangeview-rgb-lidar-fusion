import torch
import torch.nn as nn
from losses.Lovasz_Softmax import Lovasz_softmax
from losses.boundary_loss import BoundaryLoss
import time
from ioueval import *

def overfit_test(model, batchsize, rgb_sze=None, lidar_sze=None):
    model = model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-5)
    pytorch_total_params = sum(p.numel()
                               for p in model.parameters() if p.requires_grad)
    print("Number of parameters: ", pytorch_total_params / 1000000, "M")
    time_train = []
    
    evaluator = iouEval(20, device='cuda', ignore=0)
    if rgb_sze != None:
        c1, h1, w1 = rgb_sze
        input_rgb = torch.rand(batchsize, c1, h1, w1).cuda() 
        
    c2, h2, w2 = lidar_sze
    input_3D = torch.rand(batchsize, c2, h2, w2).cuda()
    label = torch.randint(0, 20, (batchsize, h2, w2)).cuda()
    criterion = nn.NLLLoss().cuda()
    ls = Lovasz_softmax(ignore=0).cuda()
    bd = BoundaryLoss().cuda()
    scaler = torch.cuda.amp.GradScaler()
    for i in range(20000):
        optimizer.zero_grad()
        model.train()
        #with torch.no_grad():
        start_time = time.time()
        with torch.cuda.amp.autocast():
            if rgb_sze == None:
                input_rgb = torch.rand(batchsize, c2, h2, w2).cuda() 
                out = model(input_3D, input_rgb)
            else:
                # ! TEST FOR FUSION ONLY
                
                if c2 == 7:
                    input_3D[:, -1, :, :] = (input_3D[:, -1, :, :] > 0.86)
                    input_3D[:, -2, :, :] = (input_3D[:, -2, :, :] > 0.3)
                else:
                    input_3D[:, -1, :, :] = (input_3D[:, -1, :, :] > 0.86)
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
                

        scaler.scale(loss_m).backward()
        scaler.step(optimizer)
        scaler.update()
        
        evaluator.addBatch(output.argmax(dim=1), label)
        accuracy = evaluator.getacc()
        jaccard, _ = evaluator.getIoUMissingClass()
        
        fwt = time.time() - start_time
        time_train.append(fwt)
        
        print("Loss: {:.2f}".format(loss_m.mean().item()), "Accuracy: {:.2f}".format(accuracy.item()), "IoU: {:.2f}".format(jaccard.item()), "Time: {:.2f}".format(fwt))
