from statistics import mean
import torch
from torch import nn
import torch.nn.functional as F
from parse_args import args
import numpy as np
import math
from Utils.Utils import *
import dgl.function as fn

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform
from torch.nn.init import xavier_uniform_

device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')

import torch
import torch.nn as nn

class AttentionFusion(nn.Module):
    """
    Diffusion-guided Cross-Attention Fusion Module

    Inputs:
        target_embed   : [N, d]
        aux_embed      : [N, d]  (e.g. denoised auxiliary embedding)

    Output:
        fused_embed    : [N, d]
    """

    def __init__(self, embed_dim, num_heads=4, dropout=0.0):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # concat + projection
        self.fuse = nn.Linear(2 * embed_dim, embed_dim)

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, target_embed, aux_embed):
        """
        target_embed: [N, d]
        aux_embed   : [N, d]
        """

        # ---------- safety check ----------
        assert target_embed.shape == aux_embed.shape, \
            f"Shape mismatch: {target_embed.shape} vs {aux_embed.shape}"

        # ---------- cross-attention ----------
        q = target_embed.unsqueeze(0)   # [1, N, d]
        k = aux_embed.unsqueeze(0)
        v = aux_embed.unsqueeze(0)

        attn_out, _ = self.attn(q, k, v)
        attn_out = attn_out.squeeze(0)  # [N, d]

        # ---------- fusion ----------
        fused = self.fuse(
            torch.cat([target_embed, attn_out], dim=-1)
        )

        # residual + norm
        fused = self.norm(fused + target_embed)

        return fused

class HGDM(nn.Module):
    def __init__(self, d_data):
        super(HGDM, self).__init__()

        # 节点数量
        self.n_drug = d_data['drug_number']
        self.n_disease = d_data['disease_number']
        self.n_protein = d_data['protein_number']

        # 邻接矩阵（不同维度）
        self.target_adj = d_data['drdi_adj']  # (n_drug + n_disease, n_drug + n_disease)
        self.drpr_adj = d_data['drpr_adj']  # (n_drug + n_protein, n_drug + n_protein)
        self.dipr_adj = d_data['dipr_adj']  # (n_disease + n_protein, n_disease + n_protein)

        self.embed_dim = args.latdim
        self.n_hid = self.embed_dim

        self.n_layers = args.gcn_layer

        # ========== ⭐ Attention Fusion Modules（参数化）==========
        self.drug_fusion = AttentionFusion(
            embed_dim=self.embed_dim,
            num_heads=args.fusion_heads,      # ← 使用参数
            dropout=args.fusion_dropout       # ← 使用参数
        )

        self.disease_fusion = AttentionFusion(
            embed_dim=self.embed_dim,
            num_heads=args.fusion_heads,
            dropout=args.fusion_dropout
        )
        # 初始化 drug/disease/protein embedding
        self.embedding_dict = self.init_weight(self.n_drug, self.n_disease, self.n_hid)

        # protein embedding
        self.protein_emb = nn.Parameter(torch.FloatTensor(self.n_protein, self.n_hid))
        nn.init.xavier_uniform_(self.protein_emb)

        self.act = nn.LeakyReLU(0.5, inplace=True)
        self.weight = False

        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(
                DGLLayer(self.n_hid, self.n_hid, weight=self.weight, bias=False, activation=self.act)
            )

        self.drpr_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.drpr_layers.append(
                DGLLayer(self.n_hid, self.n_hid, weight=self.weight, bias=False, activation=self.act)
            )

        self.dipr_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.dipr_layers.append(
                DGLLayer(self.n_hid, self.n_hid, weight=self.weight, bias=False, activation=self.act)
            )

        # 扩散模块
        self.diffusion_process = GaussianDiffusion(
            args.noise_scale, args.noise_min, args.noise_max, args.steps
        ).to(device)

        out_dims = eval(args.dims) + [args.latdim]
        in_dims = out_dims[::-1]

        self.drug_denoiser = Denoise(in_dims, out_dims, args.d_emb_size, norm=args.norm).to(device)
        self.disease_denoiser = Denoise(in_dims, out_dims, args.d_emb_size, norm=args.norm).to(device)

        self.final_act = nn.LeakyReLU(negative_slope=0.5)

    def init_weight(self, n_drug, n_disease, n_hid):
        """初始化 drug 和 disease embedding"""
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'drug_emb': nn.Parameter(initializer(torch.empty(n_drug, n_hid))),
            'disease_emb': nn.Parameter(initializer(torch.empty(n_disease, n_hid)))
        })
        return embedding_dict

    def forward(self, drdi_graph=None, drpr_graph=None, dipr_graph=None):


        # 获取初始 embeddings
        drug_emb = self.embedding_dict['drug_emb']
        disease_emb = self.embedding_dict['disease_emb']
        protein_emb = self.protein_emb

        init_embedding = torch.cat([drug_emb, disease_emb], dim=0)
        all_embeddings = [init_embedding]

        embeddings = init_embedding
        for layer in self.layers:
            embeddings = layer(
                self.target_adj,
                embeddings[:self.n_drug],  # drug 部分
                embeddings[self.n_drug:]  # disease 部分
            )
            norm_embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(norm_embeddings)

        # 聚合多层 embeddings
        dd_embeddings = sum(all_embeddings)

        drpr_init = torch.cat([drug_emb, protein_emb], dim=0)
        drpr_embeddings_list = [drpr_init]

        drpr_emb = drpr_init
        for layer in self.drpr_layers:
            drpr_emb = layer(
                self.drpr_adj,
                drpr_emb[:self.n_drug],  # drug 部分
                drpr_emb[self.n_drug:]  # protein 部分
            )
            norm_drpr_emb = F.normalize(drpr_emb, p=2, dim=1)
            drpr_embeddings_list.append(norm_drpr_emb)

        # 聚合并提取 drug 部分
        drpr_embeddings = sum(drpr_embeddings_list)
        drug_from_protein = drpr_embeddings[:self.n_drug]


        dipr_init = torch.cat([disease_emb, protein_emb], dim=0)
        dipr_embeddings_list = [dipr_init]

        dipr_emb = dipr_init
        for layer in self.dipr_layers:
            dipr_emb = layer(
                self.dipr_adj,
                dipr_emb[:self.n_disease],  # disease 部分
                dipr_emb[self.n_disease:]  # protein 部分
            )
            norm_dipr_emb = F.normalize(dipr_emb, p=2, dim=1)
            dipr_embeddings_list.append(norm_dipr_emb)

        # 聚合并提取 disease 部分
        dipr_embeddings = sum(dipr_embeddings_list)
        disease_from_protein = dipr_embeddings[:self.n_disease]

        target_drug_embedding = dd_embeddings[:self.n_drug]
        target_disease_embedding = dd_embeddings[self.n_drug:]

        return (
            target_drug_embedding,
            target_disease_embedding,
            drug_from_protein,
            disease_from_protein
        )

    def cal_loss(self, ancs, poss):
        """

        Args:
            ancs: [batch] 药物 ID
            poss: [batch] 正例疾病 ID
            negs: [batch] 负例疾病 ID（保留参数但不使用）

        Returns:
            loss: 总损失（扩散损失 + 正则化损失）
            diff_loss: 扩散损失（用于日志）
            reg_loss: 正则化损失（用于日志）
        """
        # ========【1】前向传播，获取嵌入========
        d_drugEmbeds, d_diseaseEmbeds, h_drugEmbeds, h_diseaseEmbeds = self.forward()

        # ========【2】计算扩散损失========
        # 药物扩散损失
        u_diff_loss, diff_drugEmbeds = self.diffusion_process.training_losses2(
            self.drug_denoiser,
            d_drugEmbeds,
            h_drugEmbeds,
            ancs  # 使用采样的药物 ID
        )

        # 疾病扩散损失（使用正例疾病）
        i_diff_loss, diff_diseaseEmbeds = self.diffusion_process.training_losses2(
            self.disease_denoiser,
            d_diseaseEmbeds,
            h_diseaseEmbeds,
            poss  # 使用正例疾病 ID
        )

        # 总扩散损失
        diff_loss = u_diff_loss.mean() + i_diff_loss.mean()

        # ========【3】融合扩散增强的嵌入========
        d_drugEmbeds = d_drugEmbeds + diff_drugEmbeds
        d_diseaseEmbeds = d_diseaseEmbeds + diff_diseaseEmbeds

        # ========【4】提取当前 batch 的嵌入========
        ancEmbeds = d_drugEmbeds[ancs]  # [batch, latdim]
        posEmbeds = d_diseaseEmbeds[poss]  # [batch, latdim]

        # ========【5】计算正则化损失（L2 范数）】========

        regLoss = (
                          (torch.norm(ancEmbeds) ** 2 + torch.norm(posEmbeds) ** 2) * args.reg
                  ) / args.batch

        # ========【6】总损失 = 扩散损失 + 正则化损失========
        loss = diff_loss + regLoss

        # ========【7】返回损失（供日志记录）】========
        return loss, diff_loss, regLoss

    def predict(self,drdi_graph=None, drpr_graph=None, dipr_graph=None):

            d_drugEmbeds, d_diseaseEmbeds, h_drugEmbeds, h_diseaseEmbeds = \
                self.forward(drdi_graph, drpr_graph, dipr_graph)

            # 2. diffusion denoising
            denoised_d = self.diffusion_process.p_sample(
                self.drug_denoiser,
                h_drugEmbeds,
                args.sampling_steps
            )

            denoised_di = self.diffusion_process.p_sample(
                self.disease_denoiser,
                h_diseaseEmbeds,
                args.sampling_steps
            )

            # 3. attention fusion
            d_drugEmbeds = self.drug_fusion(d_drugEmbeds, denoised_d)
            d_diseaseEmbeds = self.disease_fusion(d_diseaseEmbeds, denoised_di)

            return d_drugEmbeds, d_diseaseEmbeds




class DGLLayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 weight=False,
                 bias=False,
                 activation=None):
        super(DGLLayer, self).__init__()
        self.bias = bias
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.weight = weight
        if self.weight:
            self.drug_w = nn.Parameter(torch.Tensor(in_feats, out_feats))
            self.disease_w = nn.Parameter(torch.Tensor(in_feats, out_feats))
            xavier_uniform_(self.drug_w)
            xavier_uniform_(self.disease_w)
        self._activation = activation

    def forward(self, graph, drug_f, disease_f):

        # ========【1】线性变换（可选）========
        if self.weight:
            drug_f = torch.mm(drug_f, self.drug_w)
            disease_f = torch.mm(disease_f, self.disease_w)

        # ========【2】合并节点特征========
        node_f = torch.cat([drug_f, disease_f], dim=0)

        # ========【3】判断输入类型========
        if isinstance(graph, torch.Tensor):
            # 稀疏邻接矩阵处理（原有逻辑）
            if graph.is_sparse:
                # 计算度
                degs = torch.sparse.sum(graph, dim=1).to_dense().clamp(min=1)
            else:
                degs = graph.sum(dim=1).clamp(min=1)

            # 度归一化
            norm = torch.pow(degs, -0.5).unsqueeze(1)
            node_f = node_f * norm

            # 消息传递
            if graph.is_sparse:
                node_f = torch.sparse.mm(graph, node_f)
            else:
                node_f = torch.mm(graph, node_f)

            node_f = node_f * norm

        else:
            # DGL 图处理（如果需要兼容）
            with graph.local_scope():
                # 度归一化
                degs = graph.out_degrees().to(node_f.device).float().clamp(min=1)
                norm = torch.pow(degs, -0.5).view(-1, 1)
                node_f = node_f * norm

                # 消息传递
                graph.ndata['n_f'] = node_f
                graph.update_all(
                    fn.copy_u(u='n_f', out='m'),
                    reduce_func=fn.sum(msg='m', out='n_f')
                )
                node_f = graph.ndata['n_f']

                # 入度归一化
                degs = graph.in_degrees().to(node_f.device).float().clamp(min=1)
                norm = torch.pow(degs, -0.5).view(-1, 1)
                node_f = node_f * norm

        # ========【4】激活函数========
        if self._activation is not None:
            node_f = self._activation(node_f)

        return node_f


class Denoise(nn.Module):


    def __init__(self, in_dims, out_dims, emb_size, norm=False, dropout=0.5):
        super(Denoise, self).__init__()

        # 输入维度列表，每一层的输入维度
        self.in_dims = in_dims
        # 输出维度列表，每一层的输出维度
        self.out_dims = out_dims
        # time embedding 的维度（用于扩散时间编码）
        self.time_emb_dim = emb_size
        # 是否归一化
        self.norm = norm

        # ========【1】time embedding 线性层========
        # 将时间编码映射到和 embedding 维度一致
        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        # ========【2】构建输入层线性层序列========
        # 第 0 层输入维度加上 time embedding
        in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]

        out_dims_temp = self.out_dims

        # 输入层列表
        self.in_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])]
        )

        # 输出层列表
        self.out_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])]
        )

        # Dropout 层
        self.drop = nn.Dropout(dropout)

        # 初始化权重
        self.init_weights()

    # ========【3】初始化权重函数========
    def init_weights(self):
        # 输入层权重初始化
        for layer in self.in_layers:
            size = layer.weight.size()
            std = np.sqrt(2.0 / (size[0] + size[1]))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)

        # 输出层权重初始化
        for layer in self.out_layers:
            size = layer.weight.size()
            std = np.sqrt(2.0 / (size[0] + size[1]))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)

        # time embedding 层初始化
        size = self.emb_layer.weight.size()
        std = np.sqrt(2.0 / (size[0] + size[1]))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)

    def forward(self, x, timesteps, mess_dropout=True):

        # ========【1】生成时间编码（sin/cos）========
        # freqs 是频率向量，用于多尺度编码
        freqs = torch.exp(
            -math.log(10000) * torch.arange(
                start=0, end=self.time_emb_dim // 2, dtype=torch.float32
            ) / (self.time_emb_dim // 2)
        ).to(device)

        # 将时间步 t 映射到频率上
        temp = timesteps[:, None].float() * freqs[None]

        # 拼接 sin/cos 得到 time embedding
        time_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)

        # 如果维度是奇数，补零
        if self.time_emb_dim % 2:
            time_emb = torch.cat([time_emb, torch.zeros_like(time_emb[:, :1])], dim=-1)

        # ========【2】通过线性层映射时间 embedding========
        emb = self.emb_layer(time_emb)  # 输出维度和节点 embedding 相同

        # ========【3】可选对节点 embedding 做归一化========
        if self.norm:
            x = F.normalize(x)

        # ========【4】可选 dropout========
        if mess_dropout:
            x = self.drop(x)

        # ========【5】将节点 embedding 与时间 embedding 拼接========
        # 辅助图 embedding + 对应时间步 embedding
        h = torch.cat([x, emb], dim=-1)

        # ========【6】输入层 MLP========
        for i, layer in enumerate(self.in_layers):
            h = layer(h)  # 线性映射
            h = torch.tanh(h)  # 非线性激活

        # ========【7】输出层 MLP========
        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)

        # ========【8】返回 Denoiser 输出========
        # 输出 h 是“语义增量”，最终会加到目标图 embedding 上
        return h


class GaussianDiffusion(nn.Module):
    """
    高斯扩散模块（Gaussian Diffusion）
    功能：
    - 定义扩散过程和反向扩散所需的噪声 schedule
    - 为 Denoiser 提供时间步 t 的噪声量
    """

    def __init__(self, noise_scale, noise_min, noise_max, steps, beta_fixed=True):
        super(GaussianDiffusion, self).__init__()

        # ========【1】扩散噪声参数========
        self.noise_scale = noise_scale  # 噪声缩放系数
        self.noise_min = noise_min  # 最小噪声
        self.noise_max = noise_max  # 最大噪声
        self.steps = steps  # 扩散步数

        # ========【2】历史损失记录========
        # history_num_per_term: 每步记录的历史次数
        self.history_num_per_term = 10
        # Lt_history: 每步扩散的历史损失
        self.Lt_history = torch.zeros(steps, 10, dtype=torch.float64).to(device)
        # Lt_count: 每步累计次数
        self.Lt_count = torch.zeros(steps, dtype=int).to(device)

        # ========【3】初始化噪声 schedule========
        if noise_scale != 0:
            # betas: 每一步扩散的噪声比例
            self.betas = torch.tensor(self.get_betas(), dtype=torch.float64).to(device)

            # 固定 beta 时，第一步 beta 为 0.0001 避免数值不稳定
            if beta_fixed:
                self.betas[0] = 0.0001

            # 计算扩散过程所需的 alpha, alpha_bar 等值
            self.calculate_for_diffusion()

    # ========【4】生成 beta 列表========
    def get_betas(self):
        """
        功能：生成扩散过程每步的噪声 beta
        beta 决定每一步添加多少噪声

        步骤：
        1. 根据 noise_scale, noise_min, noise_max 生成方差序列
        2. 计算 alpha_bar（累计保留信号比例）
        3. 计算每步 beta，保证不会超过 0.999
        """
        start = self.noise_scale * self.noise_min
        end = self.noise_scale * self.noise_max

        # 线性方差序列
        variance = np.linspace(start, end, self.steps, dtype=np.float64)

        # alpha_bar: 每步保留信号比例
        alpha_bar = 1 - variance

        # betas: 每步添加噪声比例
        betas = []
        betas.append(1 - alpha_bar[0])  # 第一步 beta

        for i in range(1, self.steps):
            # 防止 beta 太大，最大 0.999
            betas.append(min(1 - alpha_bar[i] / alpha_bar[i - 1], 0.999))

        return np.array(betas)

    def calculate_for_diffusion(self):
        """
        计算扩散所需的各种中间量：
        - alpha, alpha_bar
        - 后验均值和方差
        这些在扩散过程和反向去噪过程中会用到
        """

        # 每步保留信号比例
        alphas = 1.0 - self.betas

        # 累乘得到 alpha_bar：t 步的累计保留信号
        self.alphas_cumprod = torch.cumprod(alphas, axis=0).to(device)
        # alpha_bar 的前一时刻
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(device), self.alphas_cumprod[:-1]]).to(device)
        # alpha_bar 的下一时刻
        self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0]).to(device)]).to(device)

        # 一些常用 sqrt/recip 预计算
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # 后验方差 (posterior variance)
        self.posterior_variance = (
                self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        # log 后验方差，防止 log(0)
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]])
        )

        # 后验均值系数，用于反向采样
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = (
                    (1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod))

    def p_sample(self, model, x_start, steps):

        # ========【1】初始化 x_t========
        if steps == 0:
            x_t = x_start  # 如果步数为 0，直接返回原始 embedding
        else:
            # 从最后一步噪声开始
            t = torch.tensor([steps - 1] * x_start.shape[0]).to(device)
            x_t = self.q_sample(x_start, t)  # 添加噪声到 t 步

        # ========【2】反向循环采样========
        # 从最后一步开始逆序采样
        indices = list(range(self.steps))[::-1]

        for i in indices:
            t = torch.tensor([i] * x_t.shape[0]).to(device)
            # 计算当前步的后验均值和方差
            model_mean, model_log_variance = self.p_mean_variance(model, x_t, t)
            # 取后验均值作为下一步输入（去噪后的 embedding）
            x_t = model_mean

        # ========【3】输出去噪后的 embedding========
        return x_t

    def q_sample(self, x_start, t, noise=None):

        if noise is None:
            # 生成与 x_start 同形状的高斯噪声
            noise = torch.randn_like(x_start)

        # x_t = sqrt(alpha_bar_t) * x_start + sqrt(1 - alpha_bar_t) * noise
        return self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + \
            self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        """
        根据时间步从数组中提取对应值，并扩展到指定形状
        用于 alpha_bar, posterior_variance 等参数的广播
        """
        arr = arr.to(device)
        # 提取对应时间步的值
        res = arr[timesteps].float()
        # 如果维度不足，增加维度
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        # 扩展到与 x 相同的形状，便于逐元素计算
        return res.expand(broadcast_shape)

    def p_mean_variance(self, model, x, t):
        """
        计算反向采样的后验均值和方差
        输入：
            model: Denoiser
            x: 当前时间步的 embedding（辅助图）
            t: 当前时间步
        输出：
            model_mean: 当前步的后验均值（去噪后的 embedding）
            model_log_variance: 当前步的后验 log 方差
        """

        # Denoiser 对当前 embedding 添加时间编码进行去噪
        model_output = model(x, t, mess_dropout=False)

        # 后验方差与 log 方差
        model_variance = self.posterior_variance
        model_log_variance = self.posterior_log_variance_clipped

        # 根据时间步提取对应方差值，并扩展到 x 的形状
        model_variance = self._extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)

        # 后验均值 = coef1 * model_output + coef2 * x
        model_mean = (self._extract_into_tensor(self.posterior_mean_coef1, t, x.shape) * model_output +
                      self._extract_into_tensor(self.posterior_mean_coef2, t, x.shape) * x)

        return model_mean, model_log_variance

    def training_losses(self, model, targetEmbeds, x_start):

        batch_size = x_start.size(0)
        # 随机选择每个样本的扩散时间步 t
        ts = torch.randint(0, self.steps, (batch_size,)).long().to(device)
        # 生成随机噪声
        noise = torch.randn_like(x_start)

        # ========【1】正向扩散：加入噪声========
        if self.noise_scale != 0:
            # 添加噪声到输入 embedding
            x_t = self.q_sample(targetEmbeds, ts, noise)
        else:
            x_t = x_start

        # ========【2】Denoiser 网络去噪========
        model_output = model(x_t, ts)

        # ========【3】计算均方误差 loss========
        mse = self.mean_flat((targetEmbeds - model_output) ** 2)

        # ========【4】加权 SNR 以调整不同时间步的 loss========
        weight = self.SNR(ts - 1) - self.SNR(ts)
        weight = torch.where((ts == 0), 1.0, weight)

        # ========【5】最终加权 diff_loss========
        diff_loss = weight * mse

        return diff_loss, model_output

    def training_losses2(self, model, targetEmbeds, x_start, batch):

        batch_size = x_start.size(0)
        device = x_start.device

        # 随机选择每个样本的扩散时间步
        ts = torch.randint(0, self.steps, (batch_size,)).long().to(device)
        noise = torch.randn_like(x_start)

        # ========【1】正向扩散========
        if self.noise_scale != 0:
            x_t = self.q_sample(x_start, ts, noise)  # 对辅助图 embedding 加噪
        else:
            x_t = x_start

        # ========【2】Denoiser 去噪========
        model_output = model(x_t, ts)

        # ========【3】均方误差 loss========
        mse = self.mean_flat((targetEmbeds - model_output) ** 2)

        # ========【4】加权 SNR========
        weight = self.SNR(ts - 1) - self.SNR(ts)
        weight = torch.where((ts == 0), 1.0, weight)

        diff_loss = weight * mse

        # ========【5】只保留 batch 对应样本========
        diff_loss = diff_loss[batch]

        return diff_loss, model_output

    def mean_flat(self, tensor):
        """
        对除第 0 维之外的所有维度求均值
        """
        return tensor.mean(dim=list(range(1, len(tensor.shape))))

    def SNR(self, t):
        """
        计算信噪比 SNR(t) = alpha_bar / (1 - alpha_bar)
        alpha_bar 是扩散正向累乘保留比例
        """
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])

    def sample_timesteps(self, batch_size, device, method='uniform', uniform_prob=0.001):
        """
        对每个样本采样一个扩散时间步 t，用于正向扩散噪声加入或 Denoiser 训练。

        输入:
            batch_size: 当前 batch 的样本数量
            device: 计算设备（CPU/GPU）
            method: 时间步采样方法
                - 'uniform': 均匀随机采样
                - 'importance': 基于历史损失 Lt 的重要性采样
            uniform_prob: importance sampling 时的平滑概率
        输出:
            t: 每个样本对应的时间步 t
            pt: 每个时间步对应的采样概率（importance sampling 使用）
        """

        if method == 'importance':  # ========【1】重要性采样========
            # 如果历史损失统计尚未满，退回到均匀采样
            if not (self.Lt_count == self.history_num_per_term).all():
                return self.sample_timesteps(batch_size, device, method='uniform')

            # 计算每个时间步的平均历史损失（平方根）
            Lt_sqrt = torch.sqrt(torch.mean(self.Lt_history ** 2, axis=-1))
            # 归一化得到每个时间步的概率
            pt_all = Lt_sqrt / torch.sum(Lt_sqrt)
            # 平滑处理，避免概率为 0
            pt_all *= 1 - uniform_prob
            pt_all += uniform_prob / len(pt_all)

            assert pt_all.sum(-1) - 1. < 1e-5  # 保证概率和为 1

            # 按概率采样 batch_size 个时间步
            t = torch.multinomial(pt_all, num_samples=batch_size, replacement=True)
            # 对采样的时间步获取对应概率
            pt = pt_all.gather(dim=0, index=t) * len(pt_all)

            return t, pt

        elif method == 'uniform':  # ========【2】均匀采样========
            # 均匀随机采样时间步 t
            t = torch.randint(0, self.steps, (batch_size,), device=device).long()
            # 每个时间步概率为 1（不做 importance weighting）
            pt = torch.ones_like(t).float()

            return t, pt

        else:
            raise ValueError
