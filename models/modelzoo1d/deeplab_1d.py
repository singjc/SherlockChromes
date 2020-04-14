import torch
import torch.nn as nn

class Conv1dResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dilation=1,
        bias=False,
        dropout=0.0,
        res=True
    ):
        super(Conv1dResBlock, self).__init__()

        padding = ((kernel_size - 1) // 2 * dilation)
        self.res = res

        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=bias
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        if self.res:
            self.residual = (
                nn.Identity() if in_channels == out_channels else nn.Conv1d(
                    in_channels,
                    out_channels,
                    1,
                    bias=False
                )
            )

    def forward(self, x):
        out = self.block(x)
        
        if self.res:
            out+= self.residual(x)

        return out

class ASPP1d(nn.Module):
    def __init__(self, n_filters=32, dilations=[6, 12, 18, 24], dropout=0.5):
        super(ASPP1d, self).__init__()

        self.aspp1 = Conv1dResBlock(
            in_channels=n_filters,
            out_channels=n_filters,
            kernel_size=1,
            dilation=1,
            bias=False,
            res=False
        )
        self.aspp2 = Conv1dResBlock(
            in_channels=n_filters,
            out_channels=n_filters,
            kernel_size=3,
            dilation=dilations[0],
            bias=False,
            res=False
        )
        self.aspp3 = Conv1dResBlock(
            in_channels=n_filters,
            out_channels=n_filters,
            kernel_size=3,
            dilation=dilations[1],
            bias=False,
            res=False
        )
        self.aspp4 = Conv1dResBlock(
            in_channels=n_filters,
            out_channels=n_filters,
            kernel_size=3,
            dilation=dilations[2],
            bias=False,
            res=False
        )
        self.aspp5 = Conv1dResBlock(
            in_channels=n_filters,
            out_channels=n_filters,
            kernel_size=3,
            dilation=dilations[3],
            bias=False,
            res=False
        )
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(n_filters, n_filters, 1, bias=False),
            nn.BatchNorm1d(n_filters),
            nn.ReLU()
        )

    def forward(self, x):
        out1 = self.aspp1(x)
        out2 = self.aspp2(x)
        out3 = self.aspp3(x)
        out4 = self.aspp4(x)
        out5 = self.aspp5(x)
        out6 = self.global_avg_pool(x).expand_as(output1)

        out = torch.cat([out1, out2, out3, out4, out5, out6], dim=1)

        return out

class DeepLab1d(nn.Module):
    def __init__(self, in_channels=6, n_filters=32, dropout_h=0.0, res=False):
        super(DeepLab1d, self).__init__()

        self.encoder = nn.ModuleList()
        
        self.encoder.append(
            Conv1dResBlock(
                in_channels=in_channels,
                out_channels=n_filters,
                kernel_size=3,
                dilation=1,
                bias=False,
                dropout=dropout_h,
                res=res
            )
        )

        for i in range(1, 4):
            self.encoder.append(
                Conv1dResBlock(
                    in_channels=n_filters,
                    out_channels=n_filters,
                    kernel_size=3,
                    dilation=1,
                    bias=False,
                    dropout=dropout_h,
                    res=res
                )
            )

        self.encoder = nn.Sequential(*self.encoder)

        self.aspp = ASPP1d(n_filters=n_filters, dilations=[6, 12, 18, 24])

        self.decoder = nn.Sequential(
            Conv1dResBlock(
                in_channels=6 * n_filters,
                out_channels=n_filters,
                kernel_size=1,
                dilation=1,
                bias=False,
                dropout=0.5,
                res=False
            ),
            Conv1dResBlock(
                Conv1dResBlock(
                    in_channels=n_filters,
                    out_channels=n_filters // 2,
                    kernel_size=3,
                    dilation=1,
                    bias=False,
                    dropout=0.5,
                    res=False
                )
            ),
            Conv1dResBlock(
                Conv1dResBlock(
                    in_channels=n_filters // 2,
                    out_channels=1,
                    kernel_size=1,
                    dilation=1,
                    bias=False,
                    dropout=0.1,
                    res=False
                )
            ),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.encoder(x)
        out = self.aspp(out)
        out = self.decoder(out)

        return out