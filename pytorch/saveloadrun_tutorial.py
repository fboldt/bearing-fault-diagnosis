import torch
import torchvision.models as models

model = models.vgg16(weights='IMAGENET1K_V1')
model_name = 'pytorch/model_weights.pth'
torch.save(model.state_dict(), model_name)

model = models.vgg16() # we do not specify ``weights``, i.e. create untrained model
model.load_state_dict(torch.load(model_name))
print(model.eval())
