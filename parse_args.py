import argparse


def ParseArgs():
    parser = argparse.ArgumentParser(description='Model Params')

    # ==================== 基础设置 ==================== #

    # ---------- 设备与随机种子 ----------
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU device id')
    parser.add_argument('--seed', type=int, default=2025,
                        help='random seed for reproducibility')

    # ---------- 数据集配置 ----------
    parser.add_argument('--dataset', type=str, default='F-dataset',
                        help='dataset name (B-dataset, C-dataset, F-dataset)')
    parser.add_argument('--data', default='C-dataset', type=str,
                        help='alias for --dataset')
    parser.add_argument('--dataset_percent', type=float, default=1.0,
                        help='percentage of dataset to use (for debugging)')
    parser.add_argument('--threshold', default=0.5, type=float,
                        help='threshold to filter associations')

    # ---------- 保存与加载 ----------
    parser.add_argument('--save_path', default='tem', type=str,
                        help='directory to save model checkpoints')
    parser.add_argument('--load_model', default=None, type=str,
                        help='path to load pretrained model')

    # ==================== 训练参数 ==================== #

    # ---------- 基础训练配置 ----------
    parser.add_argument('--epoch', default=1000, type=int,
                        help='number of training epochs')
    parser.add_argument('--total_epochs', default=1000, type=int,
                        help='total epoch number (same as --epoch)')
    parser.add_argument('--batch', default=128, type=int,
                        help='batch size for training')
    parser.add_argument('--tstBat', default=1024, type=int,
                        help='batch size for testing/validation')
    parser.add_argument('--K_fold', type=int, default=10,
                        help='k-fold cross validation')

    # ---------- 优化器参数 ----------
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='learning rate for main model (GNN)')
    parser.add_argument('--difflr', default=1e-3, type=float,
                        help='learning rate for diffusion model')
    parser.add_argument('--weight_decay', default=1e-6, type=float,
                        help='weight decay for optimizer (L2 penalty)')
    parser.add_argument('--reg', default=3e-2, type=float,
                        help='L2 regularization weight for embeddings')

    # ---------- 学习率调度 ----------
    parser.add_argument('--decay', default=0.96, type=float,
                        help='learning rate decay rate')
    parser.add_argument('--decay_step', type=int, default=1,
                        help='decay every N epochs')

    # ---------- Early Stopping ----------
    parser.add_argument('--patience', type=int, default=20,
                        help='early stopping patience (epochs)')

    # ---------- 负采样 ----------
    parser.add_argument('--negative_rate', type=float, default=1.0,
                        help='negative sampling rate (neg/pos ratio)')

    # ==================== 模型架构参数 ==================== #

    # ---------- Embedding 维度 ----------
    parser.add_argument('--latdim', default=256, type=int,
                        help='latent dimension / embedding size')#初始嵌入维度,扩散模块通常以这个维度作为输入做实验16,64,128,256,最终256
    parser.add_argument('--init', default=False, type=bool,
                        help='whether to use pretrained initial embeddings')

    # ---------- GNN 层配置 ----------
    parser.add_argument('--gcn_layer', default=2, type=int,
                        help='number of GCN layers for target graph')
    parser.add_argument('--uugcn_layer', default=2, type=int,
                        help='number of auxiliary graph GCN layers')
    parser.add_argument('--dropout', default=0.4, type=float,
                        help='dropout rate for GNN layers')
    parser.add_argument('--dropRate', default=0.5, type=float,
                        help='dropout rate (alternative name)')
    parser.add_argument('--keepRate', default=0.5, type=float,
                        help='keep rate = 1 - dropout')

    # ---------- Graph Transformer 参数 ----------
    parser.add_argument('--gt_layer', default=2, type=int,
                        help='number of graph transformer layers')
    parser.add_argument('--gt_head', default=4, type=int,
                        help='number of attention heads in graph transformer')
    parser.add_argument('--gt_out_dim', default=256, type=int,
                        help='output dimension of graph transformer')

    # ---------- Transformer 参数 ----------
    parser.add_argument('--tr_layer', default=2, type=int,
                        help='number of transformer layers')
    parser.add_argument('--tr_head', default=4, type=int,
                        help='number of attention heads in transformer')

    # ==================== 扩散模型参数 ==================== #

    # ---------- 扩散网络结构 ----------
    parser.add_argument('--difflayers', type=int, default=2,
                        help='number of layers in diffusion denoiser')
    parser.add_argument('--dims', type=str, default='[64]',
                        help='hidden dimensions for diffusion network (eval as list)')
    parser.add_argument('--d_emb_size', type=int, default=8,
                        help='time step embedding size for diffusion')
    parser.add_argument('--norm', type=bool, default=True,
                        help='whether to use normalization in diffusion network')

    # ---------- 扩散过程配置 ----------
    parser.add_argument('--steps', type=int, default=200,
                        help='total diffusion steps (T)')#总扩散步数 T,控制 DiffGraph / Diffusion Model 的前向扩散步数扩散步数做实验10 100 150 200 250
    parser.add_argument('--diffsteps', type=int, default=5,
                        help='diffusion steps during training (can be < steps)')#训练时的扩散步数
    parser.add_argument('--sampling_steps', type=int, default=0,
                        help='sampling steps during inference (0 = use all steps)')#推理时的采样步数

    # ---------- 噪声调度 ----------
    parser.add_argument('--noise_scale', type=float, default=1e-4,
                        help='noise scale factor')#控制扩散过程中高斯噪声强度,1e-3, 1e-4, 1e-5, 1e-6
    parser.add_argument('--noise_min', type=float, default=0.0001,
                        help='minimum noise level (beta_0)')
    parser.add_argument('--noise_max', type=float, default=0.001,#0.001,0.01,0.02,0.03
                        help='maximum noise level (beta_T)')

    # ---------- 扩散损失权重 ----------
    parser.add_argument('--diff_reg', type=float, default=0.00001,
                        help='weight for diffusion loss in total loss')

    # ==================== ⭐ 多头注意力融合参数 ==================== #

    parser.add_argument('--fusion_heads', type=int, default=4,
                        help='number of attention heads in fusion module')
    parser.add_argument('--fusion_dropout', type=float, default=0.0,
                        help='dropout rate for attention fusion')
    parser.add_argument('--use_fusion', type=bool, default=True,
                        help='whether to use attention fusion during inference')
    parser.add_argument('--fusion_in_train', type=bool, default=False,
                        help='whether to use attention fusion during training')

    # ==================== 对比学习参数 ==================== #

    # ---------- Intra-view SSL ----------
    parser.add_argument('--intra_ssl_temperature', type=float, default=0.05,
                        help='temperature for intra-view contrastive loss')
    parser.add_argument('--intra_ssl_reg', type=float, default=0.00001,
                        help='weight for intra-view SSL loss')

    # ---------- Inter-view SSL ----------
    parser.add_argument('--inter_ssl_temperature', type=float, default=0.05,
                        help='temperature for inter-view contrastive loss')
    parser.add_argument('--inter_ssl_reg', type=float, default=0.00001,
                        help='weight for inter-view SSL loss')

    # ---------- HGDM SSL ----------
    parser.add_argument('--ssl_reg', type=float, default=0.1,
                        help='weight for HGDM contrastive loss')
    parser.add_argument('--ssl_temp', type=float, default=0.2,
                        help='temperature for HGDM contrastive learning')

    # ==================== 图构建参数 ==================== #

    parser.add_argument('--KNN_neighbor', type=int, default=20,
                        help='number of neighbors for KNN graph construction')

    # ==================== 评估参数 ==================== #

    parser.add_argument('--topk', default=20, type=int,
                        help='top K for evaluation metrics (Recall@K, NDCG@K)')

    # ==================== 解码器参数 ==================== #

    parser.add_argument('--decoder_type', default=1, type=int,
                        help='decoder type (1: inner product, 2: MLP, etc.)')

    return parser.parse_args()

args = ParseArgs()