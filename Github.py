import torch
import torch.nn as nn
import torch.nn.functional as F

class Main_Model_SSID(nn.Module):
    def __init__(self):
        super(Main_Model_SSID, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.SP_per = EEGNet_Plus(n_classes=3, channels=124, samples=4097)
        self.SP_ima = EEGNet_Plus(n_classes=3, channels=124, samples=4097)
        self.public = Public_Part(n_classes=3, channels=124, samples=4097)

        self.conv1 = nn.Conv2d(1, 1, (1, 1))
        self.conv2 = nn.Conv2d(1, 1, (1, 1))
        self.class_pic = nn.Linear(4096, 2048)
        self.class_ima = nn.Linear(4096, 2048)
        self.capsnet = CapsuleNet([2, 1, 2048], 3, 3)
        self.loss_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        x_per_old = x[:, :124, :]
        x_img_old = x[:, 124:, :]

        x_per = self.SP_per(x_per_old)
        x_img = self.SP_ima(x_img_old)
        x_per_public = self.public(x_per_old)
        x_img_public = self.public(x_img_old)

        x_per_kl = x_per.reshape(x.size(0), -1)
        x_img_kl = x_img.reshape(x.size(0), -1)
        x_per_public_kl = x_per_public.reshape(x.size(0), -1)
        x_img_public_kl = x_img_public.reshape(x.size(0), -1)

        Per_kl = F.kl_div(
            F.log_softmax(x_per_public_kl, dim=-1),
            F.softmax(x_per_kl, dim=-1),
            reduction='batchmean'
        )
        Img_kl = F.kl_div(
            F.log_softmax(x_img_public_kl, dim=-1),
            F.softmax(x_img_kl, dim=-1),
            reduction='batchmean'
        )

        x_per_all = torch.unsqueeze(torch.cat((x_per, x_per_public), dim=1), dim=1)
        x_img_all = torch.unsqueeze(torch.cat((x_img, x_img_public), dim=1), dim=1)

        x_per_all = self.conv1(x_per_all)
        x_img_all = self.conv2(x_img_all)

        x_per_all = x_per_all.view(x_per_all.size(0), -1)
        x_img_all = x_img_all.view(x_img_all.size(0), -1)

        per_out = torch.unsqueeze(self.class_pic(x_per_all), dim=1)
        img_out = torch.unsqueeze(self.class_ima(x_img_all), dim=1)

        out = torch.unsqueeze(torch.cat((per_out, img_out), dim=1), dim=2)
        out, caps_loss = self.capsnet(out)

        return out, caps_loss, 0.1 * (0.5 * Per_kl + 0.5 * Img_kl)
