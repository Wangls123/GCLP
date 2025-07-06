import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


class PromptGenerator(nn.Module):
    def __init__(self, prompt_embeddings, gate_type, batch_size, hidden_size, num_tokens, topk=1.0, pt_depth=28):
        super().__init__()
        self.prompt_embeddings = prompt_embeddings  # shape: [pt_depth, num_tokens, hidden_size]
        self.gate_type = gate_type
        self.num_tokens = num_tokens
        self.topk = topk
        self.pt_depth = pt_depth
        self.prompt_dim = hidden_size
        self.prompt_proj = nn.Identity()
        self.timestep_embedder = nn.Sequential(
            nn.Linear(1, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size)
        )

        if gate_type == "linear":
            self.gate = nn.Linear(hidden_size + 1, num_tokens)
            self.w_noise = nn.Parameter(torch.zeros(hidden_size, num_tokens), requires_grad=True)
            nn.init.xavier_uniform_(self.gate.weight)
            nn.init.constant_(self.gate.bias, 0)
            nn.init.xavier_uniform_(self.w_noise)

    def noisy_top_k_gating(self, x, depth, train=True, noise_epsilon=1e-2):
        depth_tensor = torch.zeros((x.shape[0], 1), device=x.device) + depth  # shape: [B, 1]
        gate_input = torch.cat([x, depth_tensor], dim=-1)  # shape: [B, H+1]
        clean_logits = self.gate(gate_input)

        if train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = F.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
        else:
            noisy_logits = clean_logits

        topk = min(int(self.topk * self.num_tokens) + 1, self.num_tokens)
        top_logits, top_indices = noisy_logits.topk(topk, dim=1)
        top_k_logits = top_logits[:, :topk - 1]
        top_k_indices = top_indices[:, :topk - 1]
        top_k_gates = F.softmax(top_k_logits, dim=-1)

        zeros = torch.zeros_like(noisy_logits, dtype=top_k_gates.dtype)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        return gates  # shape: [B, N]

    def forward(self, timestep, train=True):
        step = timestep[0].item() if isinstance(timestep, torch.Tensor) else int(timestep)
        timestep = timestep.float().view(-1, 1).to(self.prompt_embeddings.device)
        # print("timestep shape:", timestep.shape)
        # print(torch.isnan(timestep).any(), torch.isinf(timestep).any())
        t_embed = self.timestep_embedder(timestep)

        # if step >= self.pt_depth:
        #     return None, t_embed

        prompt_layer = self.prompt_embeddings[step]  # shape: [N, D]
        prompts = self.prompt_proj(prompt_layer).unsqueeze(0).expand(t_embed.size(0), -1, -1)  # [B, N, D]

        if self.gate_type == "linear":
            routing_mask = self.noisy_top_k_gating(t_embed, depth=step, train=train)  # [B, N]
            prompts = prompts * routing_mask.unsqueeze(-1)  # [B, N, D]

        return prompts, t_embed

    def inject_to_embedding(self, base_embeds, prompt_embeds, mode="prepend"):
        """
        base_embeds: [B, L, D]
        prompt_embeds: [B, N, D]
        Returns: enriched [B, L+N, D] or [B, L, D] depending on mode
        """
        if prompt_embeds is None:
            return base_embeds

        if mode == "prepend":
            return torch.cat([prompt_embeds, base_embeds], dim=1)  # [B, N+L, D]
        elif mode == "add":
            length = min(base_embeds.shape[1], prompt_embeds.shape[1])
            base_embeds[:, :length, :] += prompt_embeds[:, :length, :]
            return base_embeds
        else:
            raise ValueError(f"Unknown injection mode: {mode}")


def denoise_fn_with_prompt(targets, timestep, attention_mask, bert, tokenizer, prompt_generator, mode="prepend"):
    prompt_embeds, t_embed = prompt_generator(timestep, train=bert.training)
    inputs_embeds = bert.get_input_embeddings()(targets)
    inputs_embeds = prompt_generator.inject_to_embedding(inputs_embeds, prompt_embeds, mode=mode)
    outputs = bert(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    return outputs.logits


def p_forward_with_prompt(
        denoise_fn,
        target_mask,
        x_t,
        t,
        diffusion,
        prompt_generator,
        tokenizer,
        predict_x0=True,
        return_x0=False,
        return_logits=False,
        special_case_x0=False,
        transition_probs=None,
        transition_probs_in_logits=True,
        maximum_likelihood=False,
        epsilon=1e-20,
        step_size=1,
        word_freq_logits=None
):
    logits = denoise_fn(targets=x_t, timestep=t, attention_mask=target_mask,
                        bert=denoise_fn.bert, tokenizer=tokenizer,
                        prompt_generator=prompt_generator)
    probs = torch.nn.Softmax(dim=-1)(logits)

    if not predict_x0:
        return logits if return_logits else probs

    qt_probs, _ = diffusion.sample_and_compute_posterior_q(
        x_0=probs,
        t=t - step_size,
        return_logits=return_logits,
        make_one_hot=maximum_likelihood,
        transition_probs_in_logits=transition_probs_in_logits,
        transition_probs=transition_probs,
        samples=x_t,
        epsilon=epsilon,
        step_size=step_size,
        word_freq_logits=word_freq_logits
    )

    retval_x0 = logits if return_logits else probs
    retval = qt_probs
    mask = ((t == step_size) & special_case_x0).long().view(-1, 1)
    retval = mask * retval_x0 + (1 - mask) * retval

    return (retval, retval_x0) if return_x0 else retval
