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


class MultiHeadAttentionFusion(nn.Module):
    def __init__(self, dim_visual, dim_target, dim_action, dim_out, num_heads=4):
        super(MultiHeadAttentionFusion, self).__init__()
        assert dim_out % num_heads == 0, "dim_out must be divisible by num_heads"
        self.num_heads = num_heads
        self.dim_per_head = dim_out // num_heads

        self.query = nn.Linear(dim_target, dim_out)
        self.key = nn.Linear(dim_visual, dim_out)
        self.value = nn.Linear(dim_action, dim_out)
        self.fc = nn.Linear(dim_out, dim_out)

    def forward(self, visual, target, action):
        if len(visual.shape) > 2:
            visual = visual.view(visual.size(0), -1)

        batch_size = visual.size(0)

        # Compute Q, K, V
        q = self.query(target).view(batch_size, self.num_heads, self.dim_per_head)
        k = self.key(visual).view(batch_size, self.num_heads, self.dim_per_head)
        v = self.value(action).view(batch_size, self.num_heads, self.dim_per_head)

        # Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.dim_per_head ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)  # [batch_size, num_heads, dim_per_head]

        # Concatenate heads and project
        context = context.view(batch_size, -1)  # [batch_size, dim_out]
        return F.relu(self.fc(context))


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

        # 定义视觉特征卷积层
        self.conv1 = nn.Conv2d(resnet_embedding_sz, 64, 1)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.embed_glove = nn.Linear(target_embedding_sz, 64)
        self.embed_action = nn.Linear(action_space, 10)
        pointwise_in_channels = 64 + 64 + 10
        self.pointwise = nn.Conv2d(pointwise_in_channels, 64, 1, 1)

        lstm_input_sz = 7 * 7 * 64 + 512
        self.hidden_state_sz = hidden_state_sz
        self.lstm = nn.LSTMCell(640, hidden_state_sz)

        num_outputs = action_space
        self.critic_linear = nn.Linear(hidden_state_sz, 1)
        self.actor_linear = nn.Linear(hidden_state_sz, num_outputs)

        # 初始化权重
        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain("relu")
        self.conv1.weight.data.mul_(relu_gain)
        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01
        )
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0
        )
        self.critic_linear.bias.data.fill_(0)
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
        self.action_predict_linear = nn.Linear(2 * lstm_input_sz, action_space)

        # 自适应Dropout层
        self.dropout_rate = args.dropout_rate
        self.dropout = nn.Dropout(p=self.dropout_rate)

        # 初始化模型属性
        self.detected_objects = None
        self.target_object_name = None
        np.seterr(divide='ignore')
        A_raw = torch.load("./data/gcn/adjmat.dat")
        A = normalize_adj(A_raw).tocsr().toarray()
        self.A = torch.nn.Parameter(torch.Tensor(A))
        n = int(A.shape[0])
        self.n = n

        # ResNet18特征提取
        resnet18 = models.resnet18(pretrained=True)
        modules = list(resnet18.children())[-2:]
        self.resnet18 = nn.Sequential(*modules)
        for p in self.resnet18.parameters():
            p.requires_grad = False

        # GloVe词向量
        with open("./data/gcn/objects.txt") as f:
            objects = f.readlines()
            self.objects = [o.strip() for o in objects]
        all_glove = torch.zeros(n, 300)
        glove = Glove(args.glove_file)
        for i in range(n):
            all_glove[i, :] = torch.Tensor(glove.glove_embeddings[self.objects[i]][:])
        self.all_glove = nn.Parameter(all_glove)
        self.all_glove.requires_grad = False

        # 图卷积网络层定义
        self.W0 = nn.Linear(401, 401, bias=False)
        self.W1 = nn.Linear(401, 401, bias=False)
        self.W2 = nn.Linear(401, 5, bias=False)
        self.W3 = nn.Linear(10, 1, bias=False)
        self.final_mapping = nn.Linear(n, 512)
        self.matched_object_linear = nn.Linear(self.n, 64)

        self.final_adjust = nn.Linear(512, 640)

        # 替换特征融合模块为多头注意力机制
        self.attention_fusion = MultiHeadAttentionFusion(
            dim_visual=3136,
            dim_target=64,
            dim_action=10,
            dim_out=64,
            num_heads=4  # 指定多头数
        )

        # 高阶特征投影模块
        self.high_dimensional_projection = HighDimensionalProjection(
            input_dim=640,
            output_dim=args.hidden_state_sz
        )

        self.visual_projection = nn.Linear(3136, 512)

    def adjust_dropout(self, epoch, max_epochs):
        # 动态调整Dropout的概率 (修改4)
        self.dropout_rate = max(0.1, self.dropout_rate * (1 - epoch / max_epochs))
        self.dropout.p = self.dropout_rate

    def is_target_in_view(self, objbb, target):
        target_idx = min(torch.argmax(target).item(), len(self.objects) - 1)
        target_name = self.objects[target_idx]
        return target_name in objbb

    def list_from_raw_obj(self, objbb, target):
        objstate = torch.zeros(self.n, 4)
        cos = torch.nn.CosineSimilarity(dim=1)
        glove_sim = cos(self.all_glove.detach(), target[None, :])[:, None]
        class_onehot = torch.zeros(1, self.n)
        detected_objects = []
        for k, v in objbb.items():
            if k in self.objects:
                ind = self.objects.index(k)
            else:
                continue
            class_onehot[0][ind] = 1
            objstate[ind][0] = 1
            x1 = v[0::4]
            y1 = v[1::4]
            x2 = v[2::4]
            y2 = v[3::4]
            objstate[ind][1] = np.sum(x1 + x2) / len(x1 + x2) / 300
            objstate[ind][2] = np.sum(y1 + y2) / len(y1 + y2) / 300
            objstate[ind][3] = abs(max(x2) - min(x1)) * abs(max(y2) - min(y1)) / 300 / 300
            detected_objects.append(k)
        self.detected_objects = detected_objects
        if args.gpu_ids != -1:
            objstate = objstate.cuda()
            class_onehot = class_onehot.cuda()
        objstate = torch.cat((objstate, glove_sim), dim=1)
        return objstate, class_onehot

    def new_gcn_embed(self, objstate, class_onehot):
        # 在图卷积层中引入残差连接以增强信息传播 (修改5)
        class_word_embed = torch.cat((class_onehot.repeat(self.n, 1), self.all_glove.detach()), dim=1)
        x = torch.mm(self.A, class_word_embed)
        x = F.relu(self.W0(x))
        x = torch.mm(self.A, x)
        x = F.relu(self.W1(x)) + class_word_embed  # 引入残差连接
        x = torch.mm(self.A, x)
        x = F.relu(self.W2(x))
        x = torch.cat((x, objstate), dim=1)
        x = torch.mm(self.A, x)
        x = F.relu(self.W3(x))
        x = x.view(1, self.n)
        x = self.final_mapping(x)
        return x

    def enhanced_object_search(self, current_scene):
        # 精简了高层任务序列匹配逻辑 (修改6)
        for task in task_dataset:
            if task["scene_name"] == current_scene:
                actions = task["high_pddl_actions"]
                for i, action in enumerate(actions):
                    if self.target_object_name in action["args"]:
                        if i > 0 and actions[i - 1]["args"][0] in self.detected_objects:
                            return actions[i - 1]["args"][0]
                        elif i < len(actions) - 1 and actions[i + 1]["args"][0] in self.detected_objects:
                            return actions[i + 1]["args"][0]
        return None

    def embedding(self, state, target, action_probs, objbb, matched_object_vector=None):
        state = state[None, :, :, :]
        action_embedding_input = action_probs
        glove_embedding = F.relu(self.embed_glove(target))
        glove_reshaped = glove_embedding.view(1, 64, 1, 1).repeat(1, 1, 7, 7)
        action_embedding = F.relu(self.embed_action(action_embedding_input))
        action_reshaped = action_embedding.view(1, 10, 1, 1).repeat(1, 1, 7, 7)
        image_embedding = F.relu(self.conv1(state))
        x = self.dropout(image_embedding)

        # Flatten visual features
        visual_flattened = image_embedding.view(1, -1)
        visual_projected = self.visual_projection(visual_flattened)

        # Fuse features using attention mechanism
        attention_fused = self.attention_fusion(
            visual=image_embedding.view(1, -1),  # Flattened visual feature
            target=glove_embedding,
            action=action_embedding
        )
        attention_fused = attention_fused.view(1, -1)  # Shape: [1, 64]

        # print("attention_fused shape:", attention_fused.shape)

        # Extract GCN embedding
        objstate, class_onehot = self.list_from_raw_obj(objbb, target)
        gcn_embedding = self.new_gcn_embed(objstate, class_onehot)
        gcn_embedding = gcn_embedding.view(1, -1)  # Shape: [1, 512]

        # print("gcn_embedding shape:", gcn_embedding.shape)

        # Ensure matched_object_vector exists
        if matched_object_vector is None:
            matched_object_vector = torch.zeros(self.n).to(gcn_embedding.device)
        matched_object_vector = self.matched_object_linear(matched_object_vector.view(1, -1))

        matched_object_vector = matched_object_vector.view(1, -1)  # Shape: [1, 64]
        # print("matched_object_vector shape:", matched_object_vector.shape)

        # Concatenate all features
        out = torch.cat((attention_fused, gcn_embedding, matched_object_vector), dim=1)
        # assert out.shape[1] == 640, f"Unexpected out shape: {out.shape}"
        # print("out shape before HighDimensionalProjection:", out.shape)

        # Adjust to LSTM input size
        out = self.high_dimensional_projection(out)
        out = self.final_adjust(out)

        return out, image_embedding

    def a3clstm(self, embedding, prev_hidden):
        hx, cx = self.lstm(embedding, prev_hidden)
        x = hx
        actor_out = self.actor_linear(x)
        critic_out = self.critic_linear(x)
        return actor_out, critic_out, (hx, cx)

    def forward(self, model_input, model_options):
        state = model_input.state
        objbb = model_input.objbb
        (hx, cx) = model_input.hidden
        current_scene = model_input.scene
        target = model_input.target_class_embedding
        action_probs = model_input.action_probs
        matched_object_vector = None

        if not self.is_target_in_view(objbb, target):
            matched_object = self.enhanced_object_search(current_scene)
            if matched_object:
                matched_object_idx = self.objects.index(matched_object)
                matched_object_vector = torch.zeros(self.n).to(state.device)
                matched_object_vector[matched_object_idx] = 1

        x, image_embedding = self.embedding(state, target, action_probs, objbb, matched_object_vector)
        actor_out, critic_out, (hx, cx) = self.a3clstm(x, (hx, cx))

        return ModelOutput(
            value=critic_out,
            logit=actor_out,
            hidden=(hx, cx),
            embedding=image_embedding,
        )



