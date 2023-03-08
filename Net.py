import torch
from torch import nn
from torchvision import models
class ImgEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.dim = out_dim
        self.resnet = models.resnet50(weights=None)

        fc_feature = self.resnet.fc.in_features

        self.resnet.fc = nn.Sequential(
            nn.BatchNorm1d(fc_feature * 1),
            nn.Linear(fc_feature * 1, self.dim)
        )
    def forward(self, input):
        embeddings = self.resnet(input)
        embeddings = torch.nn.functional.normalize(embeddings, p = 2, dim = 1)
        return embeddings
class ImgClsNet(nn.Module):
    def __init__(self):
        out_dim = 128
        dim1 = 64
        dim2 = 16
        p = 0.2
        super().__init__()
        self.fc1 = nn.Linear(out_dim, dim1)
        self.relu = nn.ReLU()
        self.drop_out = nn.Dropout(p = p)
        self.fc2 = nn.Linear(dim1, dim2)

    def forward(self, input):
        #要不要激活呢？
        out1 = self.relu(input)
        out1 = self.drop_out(out1)
        out1 = self.fc1(out1)

        out2 = self.relu(out1)
        out2 = self.drop_out(out2)
        out2 = self.fc2(out2)

        return out1, out2

class ImgOriNet(nn.Module):
    def __init__(self):
        out_dim = 128
        dim = 16
        p = 0.2
        super().__init__()
        self.sequential = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p = p),
            nn.Linear(out_dim, dim),
        )
    def forward(self, input):
        out = self.sequential(input)
        return out

class ImgNet(nn.Module):
    def __init__(self):
        out_dim = 128

        super().__init__()
        self.query_encoder = ImgEncoder(out_dim)
        self.clf = ImgClsNet()
        self.ori = ImgOriNet()
    def forward(self, input):
        out = self.query_encoder(input)
        o1 = self.ori(out)
        o2, o3 = self.clf(out)

        return o1,o2,o3
class ModelEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.dim = out_dim
        self.resnet = models.resnet50(weights=None)

        fc_feature = self.resnet.fc.in_features

        self.resnet.fc = nn.Sequential(
            nn.BatchNorm1d(fc_feature * 1),
            nn.Linear(fc_feature * 1, self.dim)
        )
    def forward(self, input):
        embeddings = self.resnet(input)
        embeddings = torch.nn.functional.normalize(embeddings, p = 2, dim = 1)
        return embeddings
class ModelClfNet(nn.Module):
    def __init__(self):
        out_dim = 128
        dim1 = 64
        dim2 = 16
        p = 0.2
        super().__init__()
        self.fc1 = nn.Linear(out_dim, dim1)
        self.relu = nn.ReLU()
        self.drop_out = nn.Dropout(p = p)
        self.fc2 = nn.Linear(dim1, dim2)

    def forward(self, input):
        #要不要激活呢？
        out1 = self.relu(input)
        out1 = self.drop_out(out1)
        out1 = self.fc1(out1)

        out2 = self.relu(out1)
        out2 = self.drop_out(out2)
        out2 = self.fc2(out2)

        return out1, out2
class ModelOriNet(nn.Module):
    def __init__(self):
        out_dim = 128
        dim = 16
        p = 0.2
        super().__init__()
        self.sequential = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p = p),
            nn.Linear(out_dim, dim),
        )
    def forward(self, input):
        out = self.sequential(input)
        return out
class CrossDomNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_net = ImgNet()
        self.model_net = ModelNet()
    def forward(self, img, render):
        img = self.img_net(img)
        render = self.model_net(render)

        render_emb = self.get_embedding(img, render)

        return img, render_emb
    def get_embedding(self, img, render):
        #img: bs x dim
        #render: bs x 12 x dim
        bs, dim = img.size()
        img = img.reshape(bs, 1, dim)

        render2 = render.transpose(1,2)

        simil = torch.bmm(img, render2)
        simil = torch.squeeze(simil)

        weight = torch.softmax(simil, 1)
        weight = weight.reshape(bs, 1, dim)

        emb = torch.bmm(weight, render)

        return emb


class ModelNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.render_encoder = ModelEncoder()
        self.clf_net = ModelClfNet()
        self.ori_net = ModelOriNet()
    def forward(self, input):
        out = self.render_encoder(input)
        out1 = self.ori_net(out)
        out2, out3 = self.clf_net(out)

        return out1, out2, out3
if __name__ == '__main__':

    net1 = ImgNet().to(device='cuda')
    net2 = ModelNet().to(device='cuda')
    x1 = torch.randn(100, 8, 3, 224, 224,requires_grad=True,device='cuda')
    y1 = torch.randn(100, 8, 16,requires_grad=False,device='cuda')

    x2 = torch.randn(100, 8, 3, 224, 224,requires_grad=True, device='cuda')
    y2 = torch.randn(100, 8, 16,requires_grad=False, device='cuda')

    loss = nn.MSELoss()

    opt1 = torch.optim.Adam(net1.parameters(),lr= 0.001)
    opt2 = torch.optim.Adam(net2.parameters())

    for e in range(400):
        for i in range (100):
            o11, o12, o13 = net1(x1[i])
            o21, o22, o23 = net2(x2[i])
            l1 = loss(o13, y1[i])
            l2 = loss(o23, y2[i])

            l3 = loss(o12,o22)
            l4 = loss(o11,o21)

            l = l1 + l2 + l3 + l4
            l.backward()

            opt1.step()
            opt2.step()

            print(l.item())
