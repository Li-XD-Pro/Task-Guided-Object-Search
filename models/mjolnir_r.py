from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from utils.net_util import norm_col_init, weights_init, toFloatTensor
import scipy.sparse as sp
import numpy as np
import json
from datasets.glove import Glove
from .model_io import ModelOutput
from utils import flag_parser

args = flag_parser.parse_arguments()

def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

with open("Task_Dataset.json", 'r') as file:
    task_dataset = json.load(file)

class SingleHeadAttentionFusion(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v):
        super(SingleHeadAttentionFusion, self).__init__()
        # no projection inside: Q,K,V provided directly
        self.scale = dim_q ** 0.5

    def forward(self, q, k, v):
        # q: [B, Cq], k: [B, Ck], v: [B, Cv]
        # compute scores
        scores = torch.bmm(q.unsqueeze(1), k.unsqueeze(2)) / self.scale  # [B,1,1]
        attn = F.softmax(scores, dim=-1)
        context = torch.bmm(attn, v.unsqueeze(1))  # [B,1,Cv]
        return context.squeeze(1)

class HighDimensionalProjection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(HighDimensionalProjection, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        return self.fc(x)

class MJOLNIR_R(torch.nn.Module):
    def __init__(self, args):
        action_space = args.action_space
        target_embedding_sz = args.glove_dim
        resnet_embedding_sz = 512
        hidden_state_sz = args.hidden_state_sz
        super(MJOLNIR_R, self).__init__()

        # 视觉特征卷积 + pointwise 融合
        self.conv1 = nn.Conv2d(resnet_embedding_sz, 64, 1)
        self.maxp1 = nn.AdaptiveAvgPool2d((1,1))
        self.embed_glove = nn.Linear(target_embedding_sz, 64)
        self.embed_action = nn.Linear(action_space, 10)
        self.pointwise = nn.Conv2d(64+64+10, 64, 1)

        # LSTM & actor-critic
        lstm_input_sz = 7*7*64 + 512  # remains for other branches
        self.hidden_state_sz = hidden_state_sz
        self.lstm = nn.LSTMCell(640, hidden_state_sz)
        self.critic_linear = nn.Linear(hidden_state_sz, 1)
        self.actor_linear = nn.Linear(hidden_state_sz, action_space)

        # 权重初始化
        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain("relu")
        self.conv1.weight.data.mul_(relu_gain)
        self.actor_linear.weight.data = norm_col_init(self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        # 自适应 Dropout
        self.dropout_rate = args.dropout_rate
        self.dropout = nn.Dropout(p=self.dropout_rate)

        # GCN 相关
        np.seterr(divide='ignore')
        A_raw = torch.load("./data/gcn/adjmat.dat")
        A = normalize_adj(A_raw).tocsr().toarray()
        self.A = nn.Parameter(torch.Tensor(A))
        self.n = A.shape[0]
        resnet18 = models.resnet18(pretrained=True)
        modules = list(resnet18.children())[-2:]
        self.resnet18 = nn.Sequential(*modules)
        for p in self.resnet18.parameters(): p.requires_grad = False
        with open("./data/gcn/objects.txt") as f:
            self.objects = [o.strip() for o in f]
        glove = Glove(args.glove_file)
        all_glove = torch.zeros(self.n, 300)
        for i,o in enumerate(self.objects): all_glove[i] = torch.Tensor(glove.glove_embeddings[o])
        self.all_glove = nn.Parameter(all_glove, requires_grad=False)
        self.W0, self.W1 = nn.Linear(401,401,bias=False), nn.Linear(401,401,bias=False)
        self.W2, self.W3 = nn.Linear(401,5,bias=False), nn.Linear(10,1,bias=False)
        self.final_mapping = nn.Linear(self.n, 512)
        self.matched_object_linear = nn.Linear(self.n, 64)

        # 新注意力融合模块
        self.attention_fusion = SingleHeadAttentionFusion(dim_q=64, dim_k=512, dim_v=64)

        self.high_dimensional_projection = HighDimensionalProjection(input_dim=640, output_dim=hidden_state_sz)
        self.final_adjust = nn.Linear(hidden_state_sz, 640)

    # 省略 normalize_adj、list_from_raw_obj、new_gcn_embed、enhanced_object_search 方法，保持不变

    def embedding(self, state, target, action_probs, objbb, matched_object_vector=None):
        # 1. 图像特征
        state = state[None]
        img_feat = F.relu(self.conv1(state))      # [1,64,7,7]
        img_feat = self.maxp1(img_feat)           # [1,64,1,1]

        # 2. 目标 & 动作特征
        glove_emb = F.relu(self.embed_glove(target)).view(1,64,1,1)
        action_emb = F.relu(self.embed_action(action_probs)).view(1,10,1,1)

        # 3. 拼接 & pointwise conv
        fusion = torch.cat((img_feat, glove_emb, action_emb), dim=1)  # [1,138,1,1]
        fusion = F.relu(self.pointwise(fusion))                       # [1,64,1,1]
        fusion = fusion.view(1,64)

        # 4. GCN 特征
        objstate, class_onehot = self.list_from_raw_obj(objbb, target)
        gcn_emb = self.new_gcn_embed(objstate, class_onehot).view(1,512)

        # 5. 匹配对象特征
        if not self.is_target_in_view(objbb, target): matched_object_vector = self._get_matched_vector(objbb)
        if matched_object_vector is None:
            matched_object_vector = torch.zeros(self.n).to(gcn_emb.device)
        matched_q = self.matched_object_linear(matched_object_vector.view(1,-1))  # [1,64]

        # 6. 注意力融合: Q=matched, K=GCN, V=fusion
        attn_out = self.attention_fusion(q=matched_q, k=gcn_emb, v=fusion)      # [1,64]

        # 7. 拼接 & 投影
        out = torch.cat((attn_out, gcn_emb, fusion), dim=1)  # [1,64+512+64=640]
        out = self.high_dimensional_projection(out)
        out = self.final_adjust(out)

        return out, img_feat

    def forward(self, model_input, model_options):
        state, objbb = model_input.state, model_input.objbb
        hx, cx = model_input.hidden
        target, action_probs = model_input.target_class_embedding, model_input.action_probs
        x, img_emb = self.embedding(state, target, action_probs, objbb)
        hx, cx = self.lstm(x, (hx, cx))
        actor_out = self.actor_linear(hx)
        critic_out = self.critic_linear(hx)
        return ModelOutput(value=critic_out, logit=actor_out, hidden=(hx, cx), embedding=img_emb)
