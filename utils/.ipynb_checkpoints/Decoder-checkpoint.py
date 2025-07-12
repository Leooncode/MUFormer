import torch
import torch.nn as nn

class Endecoder(torch.nn.Module):
    def __init__(self):
        super(Endecoder, self).__init__()
        P, L = 3, 173
        self.decoder1 = nn.Sequential(
            nn.Conv2d(
                in_channels=P,
                out_channels=L,
                kernel_size=(1, 1),
                stride=(1, 1),
                bias=False,
            ),
            nn.ReLU()
        )
        
        self.decoder2 = nn.Sequential(
            nn.Conv2d(
                in_channels=P,
                out_channels=L,
                kernel_size=(1, 1),
                stride=(1, 1),
                bias=False,
            ),
            nn.ReLU()
        )
        
        self.decoder3 = nn.Sequential(
            nn.Conv2d(
                in_channels=P,
                out_channels=L,
                kernel_size=(1, 1),
                stride=(1, 1),
                bias=False,
            ),
            nn.ReLU()
        )
        
        self.decoder4 = nn.Sequential(
            nn.Conv2d(
                in_channels=P,
                out_channels=L,
                kernel_size=(1, 1),
                stride=(1, 1),
                bias=False,
            ),
            nn.ReLU()
        )
        
        self.decoder5 = nn.Sequential(
            nn.Conv2d(
                in_channels=P,
                out_channels=L,
                kernel_size=(1, 1),
                stride=(1, 1),
                bias=False,
            ),
            nn.ReLU()
        )
        self.decoder6 = nn.Sequential(
            nn.Conv2d(
                in_channels=P,
                out_channels=L,
                kernel_size=(1, 1),
                stride=(1, 1),
                bias=False,
            ),
            nn.ReLU()
        )
    
    def forward(self, endmember, out_t, P, nr1, nc1):
        out_t = out_t.view(1, P, nr1, nc1)
        # endmember
        if torch.equal(endmember, E1):
            re_out = self.decoder1(out_t)
        elif torch.equal(endmember, E2):
            re_out = self.decoder2(out_t)
        elif torch.equal(endmember, E3):
            re_out = self.decoder3(out_t)
        elif torch.equal(endmember, E4):
            re_out = self.decoder4(out_t)
        elif torch.equal(endmember, E5):
            re_out = self.decoder5(out_t)    
        elif torch.equal(endmember, E6):
            re_out = self.decoder6(out_t)
        return re_out