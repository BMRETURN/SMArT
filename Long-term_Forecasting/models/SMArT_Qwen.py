import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig


class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position):
        super(PositionalEncoding, self).__init__()
        self.register_buffer(
            "pos_table", self._get_sinusoid_encoding_table(n_position, d_hid)
        )

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        def get_position_angle_vec(position):
            return [
                position / np.power(10000, 2 * (hid_j // 2) / d_hid)
                for hid_j in range(d_hid)
            ]

        sinusoid_table = np.array(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
        )
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

        return torch.FloatTensor(sinusoid_table).view(1, n_position, d_hid)

    def forward(self):
        return self.pos_table.clone().detach()


class AttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(AttentionLayer, self).__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        x = attn_output + x
        x = self.layer_norm(x)
        return x


class MultiLayerAttention(nn.Module):
    def __init__(self, seq_len, d_model, num_layers, num_heads, dropout=0.1):
        super(MultiLayerAttention, self).__init__()
        self.position_layer = PositionalEncoding(d_model, seq_len)
        self.num_layers = num_layers

        self.attn_layers = nn.ModuleList([
            AttentionLayer(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        x = x + self.position_layer()
        for i in range(self.num_layers):
            x = self.attn_layers[i](x)
        return x


class Inception_Block(nn.Module):
    def __init__(self, in_channels, out_channels, conv=2, num_kernels=3, init_weight=True):
        super(Inception_Block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        self.conv = conv
        kernels = []
        for i in range(self.num_kernels):
            if self.conv == 1:
                kernels.append(nn.Conv1d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
            elif self.conv == 2:
                kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
            else:
                raise ValueError("conv should be 1 or 2.")

        self.kernels = nn.ModuleList(kernels)

        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class SMArT_Qwen(nn.Module):
    def __init__(self, configs, device):
        super(SMArT_Qwen, self).__init__()
        self.is_Qwen = configs.is_gpt
        self.patch_size = configs.patch_size
        self.pretrain = configs.pretrain
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num += 1
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        self.num_embeddings = 128
        self.patch_embeddings = 512
        self.device = device
        self.prompt = configs.prompt

        if configs.is_gpt:
            if configs.pretrain:
                self.qwen = AutoModel.from_pretrained(
                    "../pretrain/Qwen2.5-1.5B",
                    trust_remote_code=True,
                    output_attentions=True,
                    output_hidden_states=True
                )
            else:
                print("------------------no pretrain------------------")
                config = AutoConfig.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)
                self.qwen = AutoModel(config)

        if hasattr(self.qwen, "layers"):
            self.qwen.layers = self.qwen.layers[:configs.gpt_layers]

        self.num_layer = nn.Linear(1, self.num_embeddings)
        self.attn_layer = MultiLayerAttention(self.patch_size, self.num_embeddings, num_layers=4, num_heads=4, dropout=0.1)
        self.patch_layer = nn.Linear(configs.patch_size * self.num_embeddings, self.patch_embeddings)
        self.image_layer = nn.Sequential(
            Inception_Block(configs.in_channels, configs.out_channels, conv=2, num_kernels=configs.num_kernels),
            nn.Flatten(),
            nn.Linear(configs.out_channels * configs.patch_size * configs.patch_size, self.patch_embeddings)
        )
        self.fusion_layer = nn.Linear(self.patch_embeddings * 2, configs.d_model)
        self.out_layer = nn.Linear(configs.d_model * self.patch_num, configs.pred_len)
        self.text_prompt = torch.load('../embeddings/text_prompts_Qwen1-5B.pt')
        # self.text_prompt = torch.load('../embeddings/text_prompts_Qwen0-5B.pt')
        self.trend_layer = nn.Linear(configs.seq_len, configs.d_model)
        self.seasonal_layer = nn.Linear(configs.seq_len, configs.d_model)

        if configs.freeze and configs.pretrain:
            for i, (name, param) in enumerate(self.qwen.named_parameters()):
                if 'ln' in name or 'wpe' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        for layer in (self.qwen, self.num_layer, self.patch_layer, self.attn_layer, self.image_layer, self.fusion_layer,
                      self.out_layer, self.trend_layer, self.seasonal_layer):
            layer.to(device=device)
            layer.train()

    def gramian_angular_field(self, x, method='difference'):
        """ Convert a patch into Gramian Angular Field (GAF) """
        x = (x - x.min(dim=-1, keepdim=True)[0]) / (
                x.max(dim=-1, keepdim=True)[0] - x.min(dim=-1, keepdim=True)[0] + 1e-5)
        phi = torch.arccos(x)

        if method == 'difference':
            gaf = torch.cos(phi.unsqueeze(-2) + phi.unsqueeze(-1))
        elif method == 'summation':
            gaf = torch.sin(phi.unsqueeze(-2) - phi.unsqueeze(-1))
        else:
            raise ValueError(f"Method '{method}' is not recognized.")
        return gaf

    def frequency_decomposition(self, x, low_cutoff=0.1):
        """ Convert a time series into Frequency Field (FF) """
        B, L = x.shape
        fft_coeff = torch.fft.fft(x, dim=-1)
        cutoff = int(low_cutoff * L)
        # low-frequency
        low_freq_coeff = fft_coeff.clone()
        low_freq_coeff[:, cutoff:-cutoff] = 0
        trend = torch.fft.ifft(low_freq_coeff, dim=-1).real
        # high-frequency
        high_freq_coeff = fft_coeff - low_freq_coeff
        seasonal = torch.fft.ifft(high_freq_coeff, dim=-1).real
        return trend, seasonal

    def contrastive_loss(self, x, emd_num, emd_image, temperature=0.05, eps=1e-8):
        """
        x: 原始 patch (B, P, S)  # 仅用于计算 CosSim
        emd_num: 数值模态 patch embedding (B, P, D)
        emd_image: 图像模态 patch embedding (B, P, D)
        """
        sim_matrix = F.cosine_similarity(x.unsqueeze(2),  x.unsqueeze(1),  dim=-1)  # (B, P, P)
        w = (sim_matrix + 1) / 2.0

        emd_num = F.normalize(emd_num, p=2, dim=-1)  # (B, P, D)
        emd_image = F.normalize(emd_image, p=2, dim=-1)  # (B, P, D)

        logits_N2I = torch.matmul(emd_num, emd_image.transpose(1, 2)) / temperature  # (B, P, P)
        logits_I2N = torch.matmul(emd_image, emd_num.transpose(1, 2)) / temperature  # (B, P, P)

        pos_logits_N2I = torch.diagonal(logits_N2I, dim1=1, dim2=2)  # (B, P)
        denom_N2I = torch.sum(torch.exp(w * logits_N2I), dim=-1) + eps
        loss_N2I = -torch.log(torch.exp(pos_logits_N2I) / denom_N2I)  # (B, P)
        loss_N2I = loss_N2I.mean()

        pos_logits_I2N = torch.diagonal(logits_I2N, dim1=1, dim2=2)  # (B, P)
        denom_I2N = torch.sum(torch.exp(w * logits_I2N), dim=-1) + eps
        loss_I2N = -torch.log(torch.exp(pos_logits_I2N) / denom_I2N)  # (B, P)
        loss_I2N = loss_I2N.mean()

        loss = 0.5 * (loss_N2I + loss_I2N)
        return loss

    def forward(self, x):
        B, L, M = x.shape

        # Normalization
        means = x.mean(1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x = (x - means) / stdev
        x = x.permute(0, 2, 1)
        x_all = x.reshape(B * M, L)
        trend, seasonal = self.frequency_decomposition(x_all)
        emd_trend = self.trend_layer(trend)
        emd_seasonal = self.seasonal_layer(seasonal)
        dynamic_prompt = (emd_trend + emd_seasonal).unsqueeze(1)

        # Patching
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x_num = x.reshape(B * M * self.patch_num, self.patch_size, 1)
        x_patch = x.reshape(B * M, self.patch_num, self.patch_size)

        # Numerical Processing
        emd_num = self.num_layer(x_num)
        attn_output = self.attn_layer(emd_num)
        emd_num = attn_output.reshape(B * M * self.patch_num, -1)
        emd_num = self.patch_layer(emd_num).reshape(B * M, self.patch_num, self.patch_embeddings)

        # Image Processing
        gaf = self.gramian_angular_field(x_patch, method="difference")
        gaf = gaf.reshape(B * M * self.patch_num, 1, self.patch_size, self.patch_size)
        emd_image = self.image_layer(gaf)
        emd_image = emd_image.reshape(B * M, self.patch_num, self.patch_embeddings)

        # Fusion
        contrastive_loss = self.contrastive_loss(x_patch, emd_num, emd_image)
        emd = self.fusion_layer(torch.cat([emd_num, emd_image], dim=-1))

        # Prompt
        text_prompt = self.text_prompt[self.prompt].to(self.device)
        text_prompt = text_prompt.reshape(1, 1, -1)
        text_prompt = text_prompt.expand(B * M, -1, -1)
        emd = torch.cat([text_prompt, dynamic_prompt, emd], dim=1)

        # Reasoning
        emd = self.qwen(inputs_embeds=emd).last_hidden_state[:, -self.patch_num:, :]

        outputs = self.out_layer(emd.reshape(B * M, -1))
        outputs = outputs.view(B, -1, M)

        # Inverse Normalization
        outputs = outputs * stdev
        outputs = outputs + means

        return outputs, contrastive_loss