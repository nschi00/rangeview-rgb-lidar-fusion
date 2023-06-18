import torch
import torch.nn as nn
import numpy as np
from modules.network.fast_attention import FlashMHA, FlashCrossMHA
#from flash_attn.flash_attention import FlashMHA
from flash_attn.bert_padding import pad_input, unpad_input

# Replace this with your correct GPU device
device = "cuda:0"
# x = torch.randn(6, 10, 128, dtype=torch.float16, device=device)
# y = torch.randn(6, 12, 128, dtype=torch.float16, device=device)
# mask = torch.zeros((6, 10), dtype=torch.bool, device=device)
# mask[:, :5] = True
# # a = unpad_input(x, mask)
# mha = FlashCrossMHA(128, 8, dtype=torch.float16, device=device)

# out = mha(x, y, y, key_padding_mask=mask)

# Create attention layer. This is similar to torch.nn.MultiheadAttention,
# and it includes the input and output linear layers
flash_mha = FlashMHA(
    embed_dim=128, # total channels (= num_heads * head_dim)
    num_heads=8, # number of heads
    device=device,
    dtype=torch.float16,
)
#torch_mha = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True).half().cuda()

# # Run forward pass with dummy data
x = torch.randn(
    (6, 60000, 128), # (batch, seqlen, embed_dim)
    device=device,
    dtype=torch.float16
)
mask = torch.zeros((6, 60000), device=device, dtype=torch.bool)
mask[:,:60000] = True
output = flash_mha(x,x,x, key_padding_mask=mask)
#output_2 = torch_mha(x, x, x, key_padding_mask=mask)
print(output)

# from torchvision.models import resnet50
# import torch
# from torchvision.models._utils import IntermediateLayerGetter
# return_layers = {"layer4": "out"}
# return_layers["layer3"] = "aux"

# model = resnet50(pretrained=False)
# a = torch.load('resnet50-19c8e357.pth')
# model.load_state_dict(torch.load('resnet50-19c8e357.pth'))
# model = IntermediateLayerGetter(model, return_layers=return_layers)
# a = torch.randn(1, 3, 224, 224)
# out = model(a)
# print(out)
# print(model)