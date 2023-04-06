from Model import get_resnet


ResNet = get_resnet(10)

ResNet.train()

for p in ResNet.parameters():
    if p.requires_grad:
        print(p)