from FJSP.FJSP_DRL_PPO.common_utils import nonzero_averaging
from FJSP.FJSP_DRL_PPO.model.attention_layer import *
from FJSP.FJSP_DRL_PPO.model.sub_layers import *
import torch
import torch.nn as nn
import torch.nn.functional as F


class DualAttentionNetwork(nn.Module):
    def __init__(self, config):
        """
            The implementation of dual attention network (DAN)
        :param config: a package of parameters
        """
        super(DualAttentionNetwork, self).__init__()

        self.fea_j_input_dim = config.fea_j_input_dim
        self.fea_m_input_dim = config.fea_m_input_dim
        self.output_dim_per_layer = config.layer_fea_output_dim
        self.num_heads_OAB = config.num_heads_OAB
        self.num_heads_MAB = config.num_heads_MAB
        self.last_layer_activate = nn.ELU()

        self.num_dan_layers = len(self.num_heads_OAB)
        assert len(config.num_heads_MAB) == self.num_dan_layers
        assert len(self.output_dim_per_layer) == self.num_dan_layers
        self.alpha = 0.2
        self.leaky_relu = nn.LeakyReLU(self.alpha)
        self.dropout_prob = config.dropout_prob

        num_heads_OAB_per_layer = [1] + self.num_heads_OAB
        num_heads_MAB_per_layer = [1] + self.num_heads_MAB

        # mid_dim = [self.embedding_output_dim] * (self.num_dan_layers - 1)
        mid_dim = self.output_dim_per_layer[:-1]

        j_input_dim_per_layer = [self.fea_j_input_dim] + mid_dim

        m_input_dim_per_layer = [self.fea_m_input_dim] + mid_dim

        self.op_attention_blocks = torch.nn.ModuleList()
        self.mch_attention_blocks = torch.nn.ModuleList()

        for i in range(self.num_dan_layers):
            self.op_attention_blocks.append(
                MultiHeadOpAttnBlock(
                    input_dim=num_heads_OAB_per_layer[i] * j_input_dim_per_layer[i],
                    num_heads=self.num_heads_OAB[i],
                    output_dim=self.output_dim_per_layer[i],
                    concat=True if i < self.num_dan_layers - 1 else False,
                    activation=nn.ELU() if i < self.num_dan_layers - 1 else self.last_layer_activate,
                    dropout_prob=self.dropout_prob
                )
            )

        for i in range(self.num_dan_layers):
            self.mch_attention_blocks.append(
                MultiHeadMchAttnBlock(
                    node_input_dim=num_heads_MAB_per_layer[i] * m_input_dim_per_layer[i],
                    edge_input_dim=num_heads_OAB_per_layer[i] * j_input_dim_per_layer[i],
                    num_heads=self.num_heads_MAB[i],
                    output_dim=self.output_dim_per_layer[i],
                    concat=True if i < self.num_dan_layers - 1 else False,
                    activation=nn.ELU() if i < self.num_dan_layers - 1 else self.last_layer_activate,
                    dropout_prob=self.dropout_prob
                )
            )

    def forward(self, fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx):
        """
        :param candidate: the index of candidates  [sz_b, J]
        :param fea_j: input operation feature vectors with shape [sz_b, N, 8]
        :param op_mask: used for masking nonexistent predecessors/successor
                        (with shape [sz_b, N, 3])
        :param fea_m: input operation feature vectors with shape [sz_b, M, 6]
        :param mch_mask: used for masking attention coefficients (with shape [sz_b, M, M])
        :param comp_idx: a tensor with shape [sz_b, M, M, J] used for computing T_E
                    the value of comp_idx[i, k, q, j] (any i) means whether
                    machine $M_k$ and $M_q$ are competing for candidate[i,j]
        :return:
            fea_j.shape = [sz_b, N, output_dim]
            fea_m.shape = [sz_b, M, output_dim]
            fea_j_global.shape = [sz_b, output_dim]
            fea_m_global.shape = [sz_b, output_dim]
        """
        sz_b, M, _, J = comp_idx.size()

        comp_idx_for_mul = comp_idx.reshape(sz_b, -1, J)

        for layer in range(self.num_dan_layers):
            candidate_idx = candidate.unsqueeze(-1). \
                repeat(1, 1, fea_j.shape[-1]).type(torch.int64)

            # fea_j_jc: candidate features with shape [sz_b, N, J]
            fea_j_jc = torch.gather(fea_j, 1, candidate_idx).type(torch.float32)
            comp_val_layer = torch.matmul(comp_idx_for_mul,
                                     fea_j_jc).reshape(sz_b, M, M, -1)
            fea_j = self.op_attention_blocks[layer](fea_j, op_mask)
            fea_m = self.mch_attention_blocks[layer](fea_m, mch_mask, comp_val_layer)

        fea_j_global = nonzero_averaging(fea_j)
        fea_m_global = nonzero_averaging(fea_m)

        return fea_j, fea_m, fea_j_global, fea_m_global


class OP_DANIEL(nn.Module):
    def __init__(self, config):
        """
            The implementation of the proposed learning framework for fjsp
        :param config: a package of parameters
        """
        super(OP_DANIEL, self).__init__()
        device = torch.device(config.device)

        # pair features input dim with fixed value
        self.pair_input_dim = 8

        self.embedding_output_dim = config.layer_fea_output_dim[-1]

        self.feature_exact = DualAttentionNetwork(config).to(
            device)
        self.op_agent = Operation_agent(config.num_mlp_layers_actor, 3 * self.embedding_output_dim,
                                        config.hidden_dim_actor, 5).to(device)
        self.critic = Critic(config.num_mlp_layers_critic, 2 * self.embedding_output_dim, config.hidden_dim_critic,
                             1).to(device)

        for name, p in self.named_parameters():
            if name.startswith('feature_exact.'):
                continue  # 跳过 DualAttentionNetwork 的参数
            if 'weight' in name:
                if len(p.size()) >= 2:
                    nn.init.orthogonal_(p, gain=1)
            elif 'bias' in name:
                nn.init.constant_(p, 0)

    def forward(self, fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx, dynamic_pair_mask, fea_pairs):
        """
        :param candidate: the index of candidate operations with shape [sz_b, J]
        :param fea_j: input operation feature vectors with shape [sz_b, N, 8]
        :param op_mask: used for masking nonexistent predecessors/successor
                        (with shape [sz_b, N, 3])
        :param fea_m: input operation feature vectors with shape [sz_b, M, 6]
        :param mch_mask: used for masking attention coefficients (with shape [sz_b, M, M])
        :param comp_idx: a tensor with shape [sz_b, M, M, J] used for computing T_E
                    the value of comp_idx[i, k, q, j] (any i) means whether
                    machine $M_k$ and $M_q$ are competing for candidate[i,j]
        :param dynamic_pair_mask: a tensor with shape [sz_b, J, M], used for masking
                            incompatible op-mch pairs
        :param fea_pairs: pair features with shape [sz_b, J, M, 8]
        :return:
            pi_op: scheduling policy with shape [sz_b, 4]
            pi_op: scheduling policy with shape [sz_b, 2]
            v: the value of state with shape [sz_b, 1]
        """
        # print(comp_idx.size())
        fea_j, fea_m, fea_j_global, fea_m_global = self.feature_exact(fea_j, op_mask, candidate, fea_m, mch_mask,
                                                                      comp_idx)
        sz_b, M, _, J = comp_idx.size()
        d = fea_j.size(-1)

        # collect the input of decision-making network
        candidate_idx = candidate.unsqueeze(-1).repeat(1, 1, d)
        candidate_idx = candidate_idx.type(torch.int64)

        Fea_j_JC = torch.gather(fea_j, 1, candidate_idx)

        pair_feature = fea_pairs.view(sz_b, -1, self.pair_input_dim)
        pair_feature = nonzero_averaging(pair_feature)

        mch_pool = torch.mean(fea_m, dim=1)  # [sz_b, 2*d + 8]

        Fea_Gj_input = fea_j_global.unsqueeze(1).expand_as(Fea_j_JC)
        Fea_Gm_input = fea_m_global.unsqueeze(1).repeat(1, J, 1)

        # Pair features: Aggregate over machines (mean pooling)
        fea_pairs_op = fea_pairs.view(sz_b, J, M, self.pair_input_dim).mean(dim=2)  # [sz_b, J, 8]
        # candidate_feature.shape = [sz_b, J, 2*output_dim + 8]
        op_agent_input = torch.cat((fea_j_global, fea_m_global, pair_feature), dim=-1)

        # Aggregate over jobs (mean pooling) to get [sz_b, 2*d + 8]
        op_agent_input_agg = torch.mean(op_agent_input, dim=1)  # [sz_b, 2*d + 8]

        op_scores = self.op_agent(op_agent_input)
        op_scores = op_scores.squeeze(-1)
        # mask = torch.sum(~dynamic_pair_mask, dim=-1)
        # op_scores[mask == 0] = float('-inf')
        pi_op = F.softmax(op_scores, dim=1)

        global_feature = torch.cat((fea_j_global, fea_m_global), dim=-1)
        v = self.critic(global_feature)
        return pi_op, v, fea_j, fea_m, fea_j_global, fea_m_global


class MCH_DANIEL(nn.Module):
    def __init__(self, config):
        """
            The implementation of the proposed learning framework for fjsp
        :param config: a package of parameters
        """
        super(MCH_DANIEL, self).__init__()
        device = torch.device(config.device)

        # pair features input dim with fixed value
        self.pair_input_dim = 8

        self.embedding_output_dim = config.layer_fea_output_dim[-1]

        self.mch_agent = Machine_agent(config.num_mlp_layers_actor,
                                       3 * self.embedding_output_dim,
                                       config.hidden_dim_actor, 3).to(device)

        for name, p in self.named_parameters():
            if 'weight' in name:
                if len(p.size()) >= 2:
                    nn.init.orthogonal_(p, gain=1)
            elif 'bias' in name:
                nn.init.constant_(p, 0)

    def forward(self, fea_j, fea_m, fea_j_global, fea_m_global, dynamic_pair_mask, fea_pairs, action_j):
        """
        :param candidate: the index of candidate operations with shape [sz_b, J]
        :param fea_j: input operation feature vectors with shape [sz_b, N, 8]
        :param op_mask: used for masking nonexistent predecessors/successor
                        (with shape [sz_b, N, 3])
        :param fea_m: input operation feature vectors with shape [sz_b, M, 6]
        :param mch_mask: used for masking attention coefficients (with shape [sz_b, M, M])
        :param comp_idx: a tensor with shape [sz_b, M, M, J] used for computing T_E
                    the value of comp_idx[i, k, q, j] (any i) means whether
                    machine $M_k$ and $M_q$ are competing for candidate[i,j]
        :param dynamic_pair_mask: a tensor with shape [sz_b, J, M], used for masking
                            incompatible op-mch pairs
        :param fea_pairs: pair features with shape [sz_b, J, M, 8]
        :return:
            pi_op: scheduling policy with shape [sz_b, 4]
            pi_op: scheduling policy with shape [sz_b, 2]
            v: the value of state with shape [sz_b, 1]
        """

        sz_b, J, M = dynamic_pair_mask.size()
        d = fea_j.size(-1)
        # print(action_j.shape)

        # Aggregate pair features over jobs (mean pooling)
        fea_pairs_mch = fea_pairs.view(sz_b, J, M, self.pair_input_dim).mean(dim=1)  # [sz_b, M, 8]

        action_expanded = action_j.view(sz_b, 1, 1).expand(-1, 1, d)
        action_feature = torch.gather(fea_j, 1, action_expanded)
        action_feature = action_feature.squeeze(1)
        # print(action_feature.shape)

        # Global features
        Fea_Gm_input = fea_m_global.unsqueeze(1).expand_as(fea_m)  # [sz_b, M, d]
        Fea_Gj_input = fea_j_global.unsqueeze(1).expand_as(fea_m)

        mch_agent_input = torch.cat((fea_j_global, fea_m_global, action_feature), dim=-1)  # [sz_b, M, 3*d + 8]


        # Aggregate over machines (mean pooling) to get [sz_b, 2*d + 8]
        mch_agent_input_agg = torch.mean(mch_agent_input, dim=1)  # [sz_b, 2*d + 8]

        mch_scores = self.mch_agent(mch_agent_input)  # [sz_b, 2]
        mch_scores = mch_scores.squeeze(-1)
        # action_mask = action_j.view(sz_b, 1, 1).expand(-1, 1, M)
        # mask = torch.gather(dynamic_pair_mask, 1, action_mask)
        # mask = mask.squeeze(1)
        # print(mask)
        # mch_scores[mask] = float('-inf')
        pi_mch = F.softmax(mch_scores, dim=1)

        return pi_mch
