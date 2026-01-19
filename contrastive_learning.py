import torch
import numpy as np
from parse_args import args
import torch.nn.functional as F

# 避免出现 log(0) 数值溢出
EPSILON = float(np.finfo(float).eps)


def safe_log(x):
    """
    避免 log(0) 出现 NaN，通过加上极小值 epsilon 实现安全 log
    """
    return torch.log(x + EPSILON)


def get_pos_neg_indice(drug_simlilarity_matrix, disease_simlilarity_matrix):
    """
    根据相似性矩阵生成正负样本掩码（视觉对比学习统一使用）
    Args:
        drug_simlilarity_matrix: 药物相似性矩阵 (N_drug x N_drug)
        disease_simlilarity_matrix: 疾病相似性矩阵 (N_disease x N_disease)
    Returns:
        药物正样本掩码、药物负样本掩码、疾病正样本掩码、疾病负样本掩码
    """

    drug_num = drug_simlilarity_matrix.shape[0]
    disease_num = disease_simlilarity_matrix.shape[0]

    # 正样本 = 相似性矩阵 - 自连接（对角线）
    drug_pos_indice = drug_simlilarity_matrix - np.eye(drug_num)
    disease_pos_indice = disease_simlilarity_matrix - np.eye(disease_num)

    drug_pos_indice = torch.from_numpy(drug_pos_indice).long().cuda()
    disease_pos_indice = torch.from_numpy(disease_pos_indice).long().cuda()

    # 负样本 = 相似性矩阵中为 0 的位置
    drug_neg_indice = (torch.from_numpy(drug_simlilarity_matrix).long().cuda() == 0).long()
    disease_neg_indice = (torch.from_numpy(disease_simlilarity_matrix).long().cuda() == 0).long()

    return drug_pos_indice, drug_neg_indice, disease_pos_indice, disease_neg_indice


# =======================================================================
# ⭐ ⭐ ⭐ 视内对比学习（Intra-view Contrastive Learning）
# =======================================================================

def similarity_contrastive(drug_simlilarity_matrix, disease_simlilarity_matrix,
                           drug_feature, disease_feature):
    """
    ⭐ 同一视图内的对比学习：
        - 药物-药物（drug–drug）
        - 疾病-疾病（disease–disease）
    使用语义相似性矩阵做正负样本构造。
    """

    # 获取正负样本掩码
    drug_pos_indice, drug_neg_indice, disease_pos_indice, disease_neg_indice = \
        get_pos_neg_indice(drug_simlilarity_matrix, disease_simlilarity_matrix)

    # embedding L2 归一化
    drug_feature = F.normalize(drug_feature, p=2, dim=1)
    disease_feature = F.normalize(disease_feature, p=2, dim=1)

    # reshape 用于矩阵相似度计算
    drug_feature_reshape = drug_feature.unsqueeze(1)
    disease_feature_reshape = disease_feature.unsqueeze(1)

    # 计算节点间相似度（点积）
    drug_score = torch.matmul(drug_feature_reshape, drug_feature.t()).squeeze(1)
    disease_score = torch.matmul(disease_feature_reshape, disease_feature.t()).squeeze(1)

    # 正样本
    drug_pos_score = drug_score * drug_pos_indice
    disease_pos_score = disease_score * disease_pos_indice

    # 负样本
    drug_neg_score = drug_score * drug_neg_indice
    disease_neg_score = disease_score * disease_neg_indice

    # softmax 温度缩放
    drug_pos_score = torch.exp(drug_pos_score / args.intra_ssl_temperature).sum(dim=1)
    drug_neg_score = torch.exp(drug_neg_score / args.intra_ssl_temperature).sum(dim=1)

    disease_pos_score = torch.exp(disease_pos_score / args.intra_ssl_temperature).sum(dim=1)
    disease_neg_score = torch.exp(disease_neg_score / args.intra_ssl_temperature).sum(dim=1)

    # InfoNCE
    drug_contrastive = -torch.sum(safe_log(drug_pos_score / drug_neg_score))
    disease_contrastive = -torch.sum(safe_log(disease_pos_score / disease_neg_score))

    return drug_contrastive + disease_contrastive


def inter_contrastive(drug_similarity_matrix, disease_similarity_matrix,
                      drug_feature1, disease_feature1,
                      drug_feature2, disease_feature2):
    """
    视间对比学习（Inter-view Contrastive Learning）

    目标：强制同一节点在不同视图中的嵌入保持一致
    - feature1 来自视图 A（如相似性图 GraphTransformer）
    - feature2 来自视图 B（如 HGDM 目标图+扩散）

    Args:
        drug_similarity_matrix: 药物相似性矩阵 [n_drug, n_drug]
        disease_similarity_matrix: 疾病相似性矩阵 [n_disease, n_disease]
        drug_feature1: 第一视图药物嵌入（GraphTransformer）[n_drug, dim]
        disease_feature1: 第一视图疾病嵌入（GraphTransformer）[n_disease, dim]
        drug_feature2: 第二视图药物嵌入（HGDM）[n_drug, dim]
        disease_feature2: 第二视图疾病嵌入（HGDM）[n_disease, dim]

    Returns:
        inter_loss: 视间对比学习损失
    """

    # ========【1】获取负样本掩码========
    # 视间对比只需要负样本掩码（正样本=同一节点）
    _, drug_neg_indice, _, disease_neg_indice = get_pos_neg_indice(
        drug_similarity_matrix, disease_similarity_matrix
    )

    # ========【2】L2 归一化========
    drug_feature1 = F.normalize(drug_feature1, p=2, dim=1)
    drug_feature2 = F.normalize(drug_feature2, p=2, dim=1)
    disease_feature1 = F.normalize(disease_feature1, p=2, dim=1)
    disease_feature2 = F.normalize(disease_feature2, p=2, dim=1)

    # ========【3】正样本得分（对角线：同一节点在两个视图）========
    drug_pos_score = torch.sum(drug_feature1 * drug_feature2, dim=1)
    disease_pos_score = torch.sum(disease_feature1 * disease_feature2, dim=1)

    # ========【4】负样本得分（全连接：不同节点在两个视图）========
    drug_neg_score = torch.matmul(drug_feature1, drug_feature2.t())
    disease_neg_score = torch.matmul(disease_feature1, disease_feature2.t())

    # ========【5】应用负样本掩码========
    drug_neg_score = drug_neg_indice * drug_neg_score
    disease_neg_score = disease_neg_indice * disease_neg_score

    # ========【6】温度缩放 + 指数变换========
    drug_pos_score = torch.exp(drug_pos_score / args.inter_ssl_temperature)
    drug_neg_score = torch.exp(drug_neg_score / args.inter_ssl_temperature).sum(dim=1)

    disease_pos_score = torch.exp(disease_pos_score / args.inter_ssl_temperature)
    disease_neg_score = torch.exp(disease_neg_score / args.inter_ssl_temperature).sum(dim=1)

    # ========【7】InfoNCE 损失========
    drug_ssl_loss = -torch.sum(torch.log(drug_pos_score / (drug_neg_score + 1e-10)))
    disease_ssl_loss = -torch.sum(torch.log(disease_pos_score / (disease_neg_score + 1e-10)))

    return drug_ssl_loss + disease_ssl_loss
