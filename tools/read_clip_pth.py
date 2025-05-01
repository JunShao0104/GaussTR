import torch
data = torch.load('ckpts/text_proto_embeds_clip.pth') # <class 'torch.Tensor'>
print(type(data))
print(data.shape) # torch.Size([512, 21])

