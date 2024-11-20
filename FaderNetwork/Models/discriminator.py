import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, params, mode="latent"):
        """
        Classe Discriminator unifiée pour différentes tâches :
        - "latent" : LatentDiscriminator
        - "patch" : PatchDiscriminator
        - "classifier" : Classifier
        """
        super(Discriminator, self).__init__()
        
        # Paramètres communs
        self.mode = mode
        self.img_sz = params.img_sz
        self.img_fm = params.img_fm
        self.init_fm = params.init_fm
        self.max_fm = params.max_fm
        self.hid_dim = params.hid_dim
        self.n_attr = params.n_attr
        
        if self.mode == "latent":
            self.n_layers = params.n_layers
            self.n_skip = params.n_skip
            self.dropout = params.lat_dis_dropout
            self.n_dis_layers = int(torch.log2(torch.tensor(self.img_sz)).item())
            self.conv_in_sz = self.img_sz / (2 ** (self.n_layers - self.n_skip))
            self.conv_in_fm = min(self.init_fm * (2 ** (self.n_layers - self.n_skip - 1)), self.max_fm)
            self.conv_out_fm = min(self.init_fm * (2 ** (self.n_dis_layers - 1)), self.max_fm)

            # Convolution jusqu'à une taille 1x1
            enc_layers, _ = build_layers(self.img_sz, self.img_fm, self.init_fm, self.max_fm,
                                         self.n_dis_layers, self.n_attr, 0, 'convtranspose',
                                         False, self.dropout, 0)
            self.conv_layers = nn.Sequential(*(enc_layers[self.n_layers - self.n_skip:]))
            
            # Couches entièrement connectées
            self.proj_layers = nn.Sequential(
                nn.Linear(self.conv_out_fm, self.hid_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(self.hid_dim, self.n_attr)
            )
        
        elif self.mode == "patch":
            self.n_patch_dis_layers = 3
            layers = [
                nn.Conv2d(self.img_fm, self.init_fm, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            n_in = self.init_fm
            n_out = min(2 * n_in, self.max_fm)

            for n in range(self.n_patch_dis_layers):
                stride = 1 if n == self.n_patch_dis_layers - 1 else 2
                layers += [
                    nn.Conv2d(n_in, n_out, kernel_size=4, stride=stride, padding=1),
                    nn.BatchNorm2d(n_out),
                    nn.LeakyReLU(0.2, inplace=True)
                ]
                n_in = n_out
                n_out = min(2 * n_out, self.max_fm)
            
            layers += [
                nn.Conv2d(n_out, 1, kernel_size=4, stride=1, padding=1),
                nn.Sigmoid()
            ]
            self.layers = nn.Sequential(*layers)
        
        elif self.mode == "classifier":
            self.n_clf_layers = int(torch.log2(torch.tensor(self.img_sz)).item())
            self.conv_out_fm = min(self.init_fm * (2 ** (self.n_clf_layers - 1)), self.max_fm)
            
            # Convolution jusqu'à une taille 1x1
            enc_layers, _ = build_layers(self.img_sz, self.img_fm, self.init_fm, self.max_fm,
                                         self.n_clf_layers, self.n_attr, 0, 'convtranspose',
                                         False, 0, 0)
            self.conv_layers = nn.Sequential(*enc_layers)
            
            # Couches entièrement connectées
            self.proj_layers = nn.Sequential(
                nn.Linear(self.conv_out_fm, self.hid_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(self.hid_dim, self.n_attr)
            )
        else:
            raise ValueError(f"Mode '{self.mode}' non reconnu. Utilisez 'latent', 'patch', ou 'classifier'.")

    def forward(self, x):
        if self.mode == "latent":
            assert x.size()[1:] == (self.conv_in_fm, self.conv_in_sz, self.conv_in_sz)
            conv_output = self.conv_layers(x)
            assert conv_output.size() == (x.size(0), self.conv_out_fm, 1, 1)
            return self.proj_layers(conv_output.view(x.size(0), self.conv_out_fm))
        
        elif self.mode == "patch":
            assert x.dim() == 4
            return self.layers(x).view(x.size(0), -1).mean(1).view(x.size(0))
        
        elif self.mode == "classifier":
            assert x.size()[1:] == (self.img_fm, self.img_sz, self.img_sz)
            conv_output = self.conv_layers(x)
            assert conv_output.size() == (x.size(0), self.conv_out_fm, 1, 1)
            return self.proj_layers(conv_output.view(x.size(0), self.conv_out_fm))
