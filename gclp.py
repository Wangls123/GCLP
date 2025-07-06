import torch
import torch.nn as nn
import torch.nn.functional as F
import diffusion_word_freq
import numpy as np
import copy
from utils import *
import math


class SupProtoConLoss(nn.Module):
    def __init__(self, temp):
        super().__init__()
        self.temperature = temp
        self.eps = 1e-8

    def score_func(self, x, y):
        return (1 + F.cosine_similarity(x, y, dim=-1)) / 2 + self.eps

    def forward(self, reps, labels):
        assert labels.dim() == 1, "标签应为一维"
        assert labels.min() >= 0, "标签值不能小于0"
        assert labels.max() < reps.size(1), "标签值超出类别范围"
        concated_bsz = reps.size(0)
        mask1 = labels.unsqueeze(0).expand(concated_bsz, concated_bsz)
        mask2 = labels.unsqueeze(1).expand(concated_bsz, concated_bsz)
        mask = 1 - torch.eye(concated_bsz).to(reps.device)
        pos_mask = (mask1 == mask2).long()
        rep1 = reps.unsqueeze(0).expand(concated_bsz, concated_bsz, reps.shape[-1])
        rep2 = reps.unsqueeze(1).expand(concated_bsz, concated_bsz, reps.shape[-1])
        scores = self.score_func(rep1, rep2)
        scores *= 1 - torch.eye(concated_bsz).to(scores.device)
        scores /= self.temperature
        scores -= torch.max(scores).item()
        pos_scores = scores * (pos_mask * mask)
        pos_scores = pos_scores.sum(-1) / ((pos_mask * mask).sum(-1) + self.eps)
        neg_scores = torch.exp(scores) * (1 - pos_mask)
        loss = -pos_scores + torch.log(neg_scores.sum(-1) + self.eps)
        loss_mask = (loss > 0).long()
        loss = (loss * loss_mask).sum() / (loss_mask.sum().item() + self.eps)
        return loss


def to_poincare(x, k=-1.0):
    assert torch.isfinite(x).all()
    K = 1.0 / -k
    sqrtK = K ** 0.5
    u = x[:, 0]
    v = x[:, 1:]
    denominator = u + sqrtK
    mapped_v = sqrtK * v / denominator.unsqueeze(-1)
    mapped_x = torch.cat((u.unsqueeze(-1), mapped_v), dim=-1)
    return mapped_x


class Oracle(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(Oracle, self).__init__()
        self.linear = nn.Sequential(nn.Linear(input_dim, input_dim),
                                    nn.BatchNorm1d(input_dim),
                                    nn.Dropout(0.4),
                                    nn.LeakyReLU(0.2),
                                    nn.Linear(input_dim, out_dim))
    def forward(self, x):
        return self.linear(x)


class CENET(nn.Module):
    def __init__(self, diffusion_instance, denoise_fn, num_e, num_rel, num_t, args):
        super(CENET, self).__init__()
        self.num_e = num_e
        self.num_t = num_t
        self.num_rel = num_rel
        self.args = args
        self.diffusion_instance = diffusion_instance
        self.denoise_fn = denoise_fn
        self.num_prompts = args.num_prompts
        self.prompt_dim = args.embedding_dim
        self.prompts = nn.Parameter(torch.zeros(self.num_prompts, self.prompt_dim))
        nn.init.zeros_(self.prompts)
        self.gate_net = nn.Sequential(
            nn.Linear(args.timestep_dim + 1, self.num_prompts),
            nn.Softmax(dim=-1)
        )
        self.num_blocks = getattr(args, "num_blocks", 3)
        self.rel_embeds = nn.Parameter(torch.zeros(2 * num_rel, args.embedding_dim))
        nn.init.xavier_uniform_(self.rel_embeds, gain=nn.init.calculate_gain('relu'))
        self.entity_embeds = nn.Parameter(torch.zeros(self.num_e, args.embedding_dim))
        nn.init.xavier_uniform_(self.entity_embeds, gain=nn.init.calculate_gain('relu'))
        self.linear_frequency = nn.Linear(self.num_e, args.embedding_dim)
        self.contrastive_hidden_layer = nn.Linear(3 * args.embedding_dim, args.embedding_dim)
        self.contrastive_output_layer = nn.Linear(args.embedding_dim, args.embedding_dim)
        self.oracle_layer = Oracle(3 * args.embedding_dim, 1)
        self.oracle_layer.apply(self.weights_init)
        self.linear_pred_layer_s1 = nn.Linear(2 * args.embedding_dim, args.embedding_dim)
        self.linear_pred_layer_o1 = nn.Linear(2 * args.embedding_dim, args.embedding_dim)
        self.linear_pred_layer_s2 = nn.Linear(2 * args.embedding_dim, args.embedding_dim)
        self.linear_pred_layer_o2 = nn.Linear(2 * args.embedding_dim, args.embedding_dim)
        self.weights_init(self.linear_frequency)
        self.weights_init(self.linear_pred_layer_s1)
        self.weights_init(self.linear_pred_layer_o1)
        self.weights_init(self.linear_pred_layer_s2)
        self.weights_init(self.linear_pred_layer_o2)
        self.dropout = nn.Dropout(args.dropout)
        self.logSoftmax = nn.LogSoftmax()
        self.softmax = nn.Softmax()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.crossEntropy = nn.BCELoss()
        self.oracle_mode = args.oracle_mode
        self.sup_proto_con_loss = SupProtoConLoss(temp=args.temp)
        self.load_loss_weight = args.load_loss_weight
        self.importance_loss_weight = args.importance_loss_weight
        self.gen_contrast_weight = getattr(args, "gen_contrast_weight", 0.01)
        print('CENET Initiated')

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, batch_block, mode_lk, total_data=None):
        quadruples, s_history_event_o, o_history_event_s, s_history_label_true, o_history_label_true, s_frequency, o_frequency = batch_block
        if isListEmpty(s_history_event_o) or isListEmpty(o_history_event_s):
            sub_rank, obj_rank, batch_loss = [None] * 3
            if mode_lk == 'Training':
                return batch_loss
            elif mode_lk in ['Valid', 'Test']:
                return sub_rank, batch_loss
            else:
                return None

        s = quadruples[:, 0]
        r = quadruples[:, 1]
        o = quadruples[:, 2]

        s_history_tag = copy.deepcopy(s_frequency)
        o_history_tag = copy.deepcopy(o_frequency)
        s_non_history_tag = copy.deepcopy(s_frequency)
        o_non_history_tag = copy.deepcopy(o_frequency)

        s_history_tag[s_history_tag != 0] = self.args.lambdax
        o_history_tag[o_history_tag != 0] = self.args.lambdax

        s_non_history_tag[s_history_tag == 1] = -self.args.lambdax
        s_non_history_tag[s_history_tag == 0] = self.args.lambdax

        o_non_history_tag[o_history_tag == 1] = -self.args.lambdax
        o_non_history_tag[o_history_tag == 0] = self.args.lambdax

        s_history_tag[s_history_tag == 0] = -self.args.lambdax
        o_history_tag[o_history_tag == 0] = -self.args.lambdax

        s_frequency = F.softmax(s_frequency, dim=1)
        o_frequency = F.softmax(o_frequency, dim=1)
        s_frequency_hidden = self.tanh(self.linear_frequency(s_frequency))
        o_frequency_hidden = self.tanh(self.linear_frequency(o_frequency))

        t = self.diffusion_instance.sample_t()
        block_depth = torch.arange(self.num_blocks).to(o_frequency_hidden.device)
        gate_input = torch.cat([t.unsqueeze(1).expand(-1, self.num_blocks), block_depth.unsqueeze(0)], dim=-1).float()
        weights = self.gate_net(gate_input)
        mixed_prompts = torch.einsum('bk,kd->bd', weights, self.prompts)
        self.entity_embeds.data.add_(mixed_prompts.mean(dim=1))

        if mode_lk == 'Training':
            s_nce_loss, _ = self.calculate_nce_loss(s, o, r, self.rel_embeds[:self.num_rel],
                                                    self.linear_pred_layer_s1, self.linear_pred_layer_s2,
                                                    s_history_tag, s_non_history_tag)
            o_nce_loss, _ = self.calculate_nce_loss(o, s, r, self.rel_embeds[self.num_rel:],
                                                    self.linear_pred_layer_o1, self.linear_pred_layer_o2,
                                                    o_history_tag, o_non_history_tag)
            s_spc_loss = self.calculate_spc_loss(s, r, self.rel_embeds[:self.num_rel], s_history_label_true, s_frequency_hidden)
            o_spc_loss = self.calculate_spc_loss(o, r, self.rel_embeds[self.num_rel:], o_history_label_true, o_frequency_hidden)
            nce_loss = (s_nce_loss + o_nce_loss) / 2.0
            spc_loss = (s_spc_loss + o_spc_loss) / 2.0
            reps = torch.cat((s_frequency_hidden, o_frequency_hidden), dim=0)
            labels = torch.cat((s, o), dim=0)
            if labels.max() >= reps.size(1):
                labels = torch.clamp(labels, max=reps.size(1) - 1)
            dcl_loss = self.sup_proto_con_loss(reps, labels)
            s_frequency_max, _ = torch.max(s_frequency, dim=1)
            o_frequency_max, _ = torch.max(o_frequency, dim=1)
            # s_diff_loss = self.diff_bert_loss(s, r, self.rel_embeds[:self.num_rel], s_history_label_true, s_frequency)
            # o_diff_loss = self.diff_bert_loss(o, r, self.rel_embeds[self.num_rel:], o_history_label_true, o_frequency)
            # diff_loss = (s_diff_loss + o_diff_loss) / 2.0    # 0.01 * diff_loss
            prompt_loss = self.prompt_balancing_loss(weights)
            gen_con_loss = self.generative_contrastive_loss(s, r, s_history_event_o, s_frequency_max)
            total_loss = self.args.alpha * nce_loss + (1 - self.args.alpha) * (spc_loss + dcl_loss) + 0.1 * prompt_loss + self.gen_contrast_weight * gen_con_loss
            return total_loss
        elif mode_lk in ['Valid', 'Test']:
            with torch.no_grad():
                s_history_oid = []
                o_history_sid = []
                for i in range(quadruples.shape[0]):
                    s_history_oid.append([])
                    o_history_sid.append([])
                    for con_events in s_history_event_o[i]:
                        s_history_oid[-1] += con_events[:, 1].tolist()
                    for con_events in o_history_event_s[i]:
                        o_history_sid[-1] += con_events[:, 1].tolist()
                s_nce_loss, s_preds = self.calculate_nce_loss(s, o, r, self.rel_embeds[:self.num_rel],
                                                              self.linear_pred_layer_s1, self.linear_pred_layer_s2,
                                                              s_history_tag, s_non_history_tag)
                o_nce_loss, o_preds = self.calculate_nce_loss(o, s, r, self.rel_embeds[self.num_rel:],
                                                              self.linear_pred_layer_o1, self.linear_pred_layer_o2,
                                                              o_history_tag, o_non_history_tag)
                s_ce_loss, s_pred_history_label, s_ce_all_acc = self.oracle_loss(s, r, self.rel_embeds[:self.num_rel],
                                                                                 s_history_label_true, s_frequency_hidden)
                o_ce_loss, o_pred_history_label, o_ce_all_acc = self.oracle_loss(o, r, self.rel_embeds[self.num_rel:],
                                                                                 o_history_label_true, o_frequency_hidden)
                s_mask = to_device(torch.zeros(quadruples.shape[0], self.num_e))
                o_mask = to_device(torch.zeros(quadruples.shape[0], self.num_e))
                for i in range(quadruples.shape[0]):
                    if s_pred_history_label[i].item() > 0.5:
                        s_mask[i, s_history_oid[i]] = 1
                    else:
                        s_mask[i, :] = 1
                        s_mask[i, s_history_oid[i]] = 0
                    if o_pred_history_label[i].item() > 0.5:
                        o_mask[i, o_history_sid[i]] = 1
                    else:
                        o_mask[i, :] = 1
                        o_mask[i, o_history_sid[i]] = 0
                if self.oracle_mode == 'soft':
                    s_mask = F.softmax(s_mask, dim=1)
                    o_mask = F.softmax(o_mask, dim=1)
                s_total_loss1, sub_rank1 = self.link_predict(s_nce_loss, s_preds, s_ce_loss, s, o, r, s_mask, total_data, 's', True)
                o_total_loss1, obj_rank1 = self.link_predict(o_nce_loss, o_preds, o_ce_loss, o, s, r, o_mask, total_data, 'o', True)
                batch_loss1 = (s_total_loss1 + o_total_loss1) / 2.0
                s_total_loss2, sub_rank2 = self.link_predict(s_nce_loss, s_preds, s_ce_loss, s, o, r, s_mask, total_data, 's', False)
                o_total_loss2, obj_rank2 = self.link_predict(o_nce_loss, o_preds, o_ce_loss, o, s, r, o_mask, total_data, 'o', False)
                batch_loss2 = (s_total_loss2 + o_total_loss2) / 2.0
                s_mask_gt = to_device(torch.zeros(quadruples.shape[0], self.num_e))
                o_mask_gt = to_device(torch.zeros(quadruples.shape[0], self.num_e))
                for i in range(quadruples.shape[0]):
                    if o[i] in s_history_oid[i]:
                        s_mask_gt[i, s_history_oid[i]] = 1
                    else:
                        s_mask_gt[i, :] = 1
                        s_mask_gt[i, s_history_oid[i]] = 0
                    if s[i] in o_history_sid[i]:
                        o_mask_gt[i, o_history_sid[i]] = 1
                    else:
                        o_mask_gt[i, :] = 1
                        o_mask_gt[i, o_history_sid[i]] = 0
                if self.oracle_mode == 'soft':
                    s_mask_gt = F.softmax(s_mask_gt, dim=1)
                    o_mask_gt = F.softmax(o_mask_gt, dim=1)
                s_total_loss3, sub_rank3 = self.link_predict(s_nce_loss, s_preds, s_ce_loss, s, o, r, s_mask_gt, total_data, 's', True)
                o_total_loss3, obj_rank3 = self.link_predict(o_nce_loss, o_preds, o_ce_loss, o, s, r, o_mask_gt, total_data, 'o', True)
                batch_loss3 = (s_total_loss3 + o_total_loss3) / 2.0
                return sub_rank1, obj_rank1, batch_loss1, sub_rank2, obj_rank2, batch_loss2, sub_rank3, obj_rank3, batch_loss3, (s_ce_all_acc + o_ce_all_acc) / 2
        elif mode_lk == 'Oracle':
            print('Oracle Training')
            s_ce_loss, _, _ = self.oracle_loss(s, r, self.rel_embeds[:self.num_rel], s_history_label_true, s_frequency_hidden)
            o_ce_loss, _, _ = self.oracle_loss(o, r, self.rel_embeds[self.num_rel:], o_history_label_true, o_frequency_hidden)
            reps = torch.cat((s_frequency_hidden, o_frequency_hidden), dim=0)
            labels = torch.cat((s, o), dim=0)
            if labels.max() >= reps.size(1):
                labels = torch.clamp(labels, max=reps.size(1) - 1)
            dcl_loss = self.sup_proto_con_loss(reps, labels)
            return (s_ce_loss + o_ce_loss + dcl_loss) / 2.0 + self.oracle_l1(0.01)

    def diff_bert_loss(self, actor1, r, rel_embeds, history_label, frequency):
        metrics = diffusion_word_freq.compute_kl_reverse_process(
            actor1,
            self.diffusion_instance.sample_t(),
            denoise_fn=self.denoise_fn,
            diffusion=self.diffusion_instance,
            target_mask=None,
            hybrid_lambda=1e-2,
            predict_x0=False,
            word_freq_logits=frequency
        )
        loss = metrics['loss'] / history_label.shape[0]
        return loss

    def prompt_balancing_loss(self, weights):
        load_loss = -torch.sum(weights * torch.log(weights + 1e-8)) / self.num_prompts
        importance = weights.sum(dim=0)
        importance_loss = torch.std(importance) / (torch.mean(importance) + 1e-8)
        return self.load_loss_weight * load_loss + self.importance_loss_weight * importance_loss

    def oracle_loss(self, actor1, r, rel_embeds, history_label, frequency_hidden):
        history_label_pred = torch.sigmoid(
            self.oracle_layer(torch.cat((self.entity_embeds[actor1], rel_embeds[r], frequency_hidden), dim=1)))
        tmp_label = torch.squeeze(history_label_pred).clone().detach()
        tmp_label[torch.where(tmp_label > 0.5)[0]] = 1
        tmp_label[torch.where(tmp_label < 0.5)[0]] = 0
        ce_correct = torch.sum(torch.eq(tmp_label, torch.squeeze(history_label)))
        ce_accuracy = 1. * ce_correct.item() / tmp_label.shape[0]
        print('# CE Accuracy', ce_accuracy)
        ce_loss = self.crossEntropy(torch.squeeze(history_label_pred), torch.squeeze(history_label))
        return ce_loss, history_label_pred, ce_accuracy * tmp_label.shape[0]

    def calculate_nce_loss(self, actor1, actor2, r, rel_embeds, linear1, linear2, history_tag, non_history_tag):
        preds_raw1 = self.tanh(linear1(self.dropout(torch.cat((to_poincare(self.entity_embeds[actor1]), rel_embeds[r]), dim=1))))
        preds1 = F.softmax(preds_raw1.mm(self.entity_embeds.transpose(0, 1)) + history_tag, dim=1)
        preds_raw2 = self.tanh(linear2(self.dropout(torch.cat((to_poincare(self.entity_embeds[actor1]), rel_embeds[r]), dim=1))))
        preds2 = F.softmax(preds_raw2.mm(self.entity_embeds.transpose(0, 1)) + non_history_tag, dim=1)
        nce = torch.sum(torch.gather(torch.log(preds1 + preds2), 1, actor2.view(-1, 1).long()))
        nce /= -1. * actor2.shape[0]
        pred_actor2 = torch.argmax(preds1 + preds2, dim=1)
        correct = torch.sum(torch.eq(pred_actor2, actor2))
        accuracy = 1. * correct.item() / actor2.shape[0]
        print('# Batch accuracy', accuracy)
        return nce, preds1 + preds2

    def link_predict(self, nce_loss, preds, ce_loss, actor1, actor2, r, trust_musk, all_triples, pred_known, oracle, history_tag=None, case_study=False):
        if oracle:
            preds = torch.mul(preds, trust_musk)
            print('$Batch After Oracle accuracy:', end=' ')
        else:
            print('$Batch No Oracle accuracy:', end=' ')
        pred_actor2 = torch.argmax(preds, dim=1)
        correct = torch.sum(torch.eq(pred_actor2, actor2))
        accuracy = 1. * correct.item() / actor2.shape[0]
        print(accuracy)
        total_loss = nce_loss + ce_loss
        ranks = []
        for i in range(preds.shape[0]):
            cur_s = actor1[i]
            cur_r = r[i]
            cur_o = actor2[i]
            ground = preds[i, cur_o].clone().item()
            if self.args.filtering:
                if pred_known == 's':
                    s_id = torch.nonzero(all_triples[:, 0] == cur_s).view(-1)
                    idx = torch.nonzero(all_triples[s_id, 1] == cur_r).view(-1)
                    idx = s_id[idx]
                    idx = all_triples[idx, 2]
                else:
                    s_id = torch.nonzero(all_triples[:, 2] == cur_s).view(-1)
                    idx = torch.nonzero(all_triples[s_id, 1] == cur_r).view(-1)
                    idx = s_id[idx]
                    idx = all_triples[idx, 0]
                preds[i, idx] = 0
                preds[i, cur_o] = ground
            ob_pred_comp1 = (preds[i, :] > ground).data.cpu().numpy()
            ob_pred_comp2 = (preds[i, :] == ground).data.cpu().numpy()
            ranks.append(np.sum(ob_pred_comp1) + ((np.sum(ob_pred_comp2) - 1.0) / 2) + 1)
        return total_loss, ranks

    def regularization_loss(self, reg_param):
        regularization_loss = torch.mean(self.rel_embeds.pow(2)) + torch.mean(self.entity_embeds.pow(2))
        return regularization_loss * reg_param

    def oracle_l1(self, reg_param):
        reg = 0
        for param in self.oracle_layer.parameters():
            reg += torch.sum(torch.abs(param))
        return reg * reg_param

    def nt_xent_loss(self, x, y, temperature=0.1):
        batch_size = x.shape[0]
        x_norm = F.normalize(x, dim=-1).unsqueeze(-1)
        y_norm = F.normalize(y, dim=-1).unsqueeze(-1)
        logits = torch.mm(x_norm, y_norm.T) / temperature
        labels = torch.arange(batch_size).to(x.device)
        loss = F.cross_entropy(logits, labels)
        return loss

    def generative_diffusion_generate(self, actor, r, rel_embeds, frequency, predict_x0=True, step_override=None):
        """
        利用扩散模型生成推理结果，支持指定扩散步数。
        """
        t_sample = self.diffusion_instance.sample_t() if step_override is None else \
                   torch.full((actor.shape[0],), fill_value=step_override, dtype=torch.long).to(actor.device)

        metrics = diffusion_word_freq.compute_kl_reverse_process(
            actor,
            t_sample,
            denoise_fn=self.denoise_fn,
            diffusion=self.diffusion_instance,
            target_mask=None,
            hybrid_lambda=1e-2,
            predict_x0=predict_x0,
            word_freq_logits=frequency
        )
        return metrics.get('x0', actor)


    def generative_contrastive_loss(self, s, r, s_history_event, frequency):
        """
        生成式对比学习：多步对比 (T/20, T/10, T/5, T/2)。
        """
        batch_size = s.shape[0]
        total_loss = 0.0
        T = self.diffusion_instance.num_steps
        # time_steps = [T // 20, T // 10, T // 5, T // 2]
        time_steps = [T // 10, T // 5]


        for step in time_steps:
            current_gen = self.generative_diffusion_generate(s, r, self.rel_embeds[:self.num_rel], frequency,
                                                             predict_x0=True, step_override=step)
            historical_gen_list = []
            for i in range(batch_size):
                hist_events = s_history_event[i]
                if len(hist_events) > 0:
                    gen_events = []
                    for evt in hist_events:
                        hist_obj = evt[:, 1]
                        # hist_obj_sample = hist_obj[0].unsqueeze(0)
                        hist_obj_sample = hist_obj[0].reshape(1)
                        hist_obj_sample = torch.tensor(hist_obj_sample, device=self.rel_embeds.device)
                        gen_evt = self.generative_diffusion_generate(hist_obj_sample, r[i].unsqueeze(0),
                                                                     self.rel_embeds[:self.num_rel],
                                                                     frequency[i].unsqueeze(0),
                                                                     predict_x0=True, step_override=step)
                        gen_events.append(gen_evt)
                    hist_gen = torch.mean(torch.cat(gen_events, dim=0).float(), dim=0, keepdim=True)
                else:
                    hist_gen = current_gen[i].unsqueeze(0)
                historical_gen_list.append(hist_gen)
            historical_gen = torch.cat(historical_gen_list, dim=0)
            total_loss += self.nt_xent_loss(current_gen.float(), historical_gen.float(), temperature=0.1)

        return total_loss / len(time_steps)
    
    # contrastive
    def freeze_parameter(self):
        self.rel_embeds.requires_grad_(False)
        self.entity_embeds.requires_grad_(False)
        self.linear_pred_layer_s1.requires_grad_(False)
        self.linear_pred_layer_o1.requires_grad_(False)
        self.linear_pred_layer_s2.requires_grad_(False)
        self.linear_pred_layer_o2.requires_grad_(False)
        self.linear_frequency.requires_grad_(False)
        self.contrastive_hidden_layer.requires_grad_(False)
        self.contrastive_output_layer.requires_grad_(False)

    def contrastive_layer(self, x):
        # Implement from the encoder E to the projection network P
        # x = F.normalize(x, dim=1)
        x = self.contrastive_hidden_layer(x)
        # x = F.relu(x)
        # x = self.contrastive_output_layer(x)
        # Normalize to unit hypersphere
        # x = F.normalize(x, dim=1)
        return x
    
    def calculate_spc_loss(self, actor1, r, rel_embeds, targets, frequency_hidden):
        projections = self.contrastive_layer(
            torch.cat((self.entity_embeds[actor1], rel_embeds[r], frequency_hidden), dim=1))
        targets = torch.squeeze(targets)
        """if np.random.randint(0, 10) < 1 and torch.sum(targets) / targets.shape[0] < 0.65 and torch.sum(targets) / targets.shape[0] > 0.35:
            np.savetxt("xx.tsv", projections.detach().cpu().numpy(), delimiter="\t")
            np.savetxt("yy.tsv", targets.detach().cpu().numpy(), delimiter="\t")
        """
        dot_product_tempered = torch.mm(projections, projections.T) / 1.0
        # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
        exp_dot_tempered = (
                torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
        )
        mask_similar_class = to_device(targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets)
        mask_anchor_out = to_device(1 - torch.eye(exp_dot_tempered.shape[0]))
        mask_combined = mask_similar_class * mask_anchor_out
        cardinality_per_samples = torch.sum(mask_combined, dim=1)
        log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True)))
        supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples

        supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)
        if torch.any(torch.isnan(supervised_contrastive_loss)):
            return 0
        return supervised_contrastive_loss

