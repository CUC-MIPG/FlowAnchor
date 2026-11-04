"""Spatial-aware Attention Refinement for Wan cross-attention logits."""

import torch
import numpy as np


def register_attention_control(model, modulation_config):
    """Register CA hook to DiT model for modulation only (no capture)"""

    for layer_idx, block in enumerate(model.blocks):
        original_processor = block.attn2.processor
        block.attn2.processor = CAProcessor(
            original_processor, layer_idx, modulation_config
        )


class CAProcessor:
    """Wrap original processor to modulate CA attention weights"""

    def __init__(self, original_processor, layer_idx: int, modulation_config):
        self.original_processor = original_processor
        self.layer_idx = layer_idx
        self.modulation_config = modulation_config

    def should_modulate(self):
        config = self.modulation_config
        return self.layer_idx in config.get("modulate_layers", [])

    def modulate_attention_weights(self, attn_weights):
        config = self.modulation_config
        ca_mask = config["ca_mask"]  # [1, 1, F, H, W]
        target_word_ids = config["target_word_ids"]
        row_strength = config.get("row_strength", 0.3)
        col_strength = config.get("col_strength", 0.3)

        batch, heads, seq_len, text_len = attn_weights.shape

        # Flatten mask: [F*H*W]
        mask_flat = ca_mask.squeeze(0).squeeze(0).flatten().to(attn_weights.device)

        # Token-level Modulation (paper Eq. (7))
        # For each spatial position i, modulate across text tokens (row direction)
        row_min = attn_weights.min(-1)[0].unsqueeze(-1)  # [batch, heads, seq_len, 1]
        row_max = attn_weights.max(-1)[0].unsqueeze(-1)  # [batch, heads, seq_len, 1]

        # Target tokens: pull towards row max in mask region
        for token_id in target_word_ids:
            token_attn = attn_weights[:, :, :, token_id]
            # Inside mask (M_i=1): A'_{i,j} = A_{i,j} + β₁(max_k A_{i,k} - A_{i,j})
            token_attn = token_attn + mask_flat * row_strength * (
                row_max.squeeze(-1) - token_attn
            )
            attn_weights[:, :, :, token_id] = token_attn

        # Non-target tokens: push towards row min in mask region
        non_target_mask = torch.ones(
            text_len, dtype=torch.bool, device=attn_weights.device
        )
        non_target_mask[target_word_ids] = False
        non_target_ids = torch.where(non_target_mask)[0]

        for token_id in non_target_ids:
            token_attn = attn_weights[:, :, :, token_id]
            # Inside mask (M_i=1): A'_{i,j} = A_{i,j} - β₁(A_{i,j} - min_k A_{i,k})
            token_attn = token_attn - mask_flat * row_strength * (
                token_attn - row_min.squeeze(-1)
            )
            attn_weights[:, :, :, token_id] = token_attn

        # Spatio-temporal modulation (paper Eq. (9))
        # For each target token j, modulate across spatial positions (column direction)
        for token_id in target_word_ids:
            col_values = attn_weights[:, :, :, token_id]
            col_min = col_values.min(-1)[0].unsqueeze(-1)
            col_max = col_values.max(-1)[0].unsqueeze(-1)

            # Mask region: pull towards column max
            col_values = col_values + mask_flat * col_strength * (col_max - col_values)

            # Non-mask region: push towards column min
            col_values = col_values - (1 - mask_flat) * col_strength * (
                col_values - col_min
            )

            attn_weights[:, :, :, token_id] = col_values

        return attn_weights

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        **kwargs,
    ):
        # hidden_states: [batch, seq_len, dim]
        # encoder_hidden_states: [batch, text_len, text_dim]

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        # Project to Q, K, V: [batch, seq_len, dim]
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)  # [batch, text_len, dim]
        value = attn.to_v(encoder_hidden_states)  # [batch, text_len, dim]

        # QK normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Reshape for multi-head: [batch, heads, seq_len, head_dim]
        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(
            1, 2
        )  # [batch, heads, text_len, head_dim]
        value = value.unflatten(2, (attn.heads, -1)).transpose(
            1, 2
        )  # [batch, heads, text_len, head_dim]

        # Compute attention weights: [batch, heads, seq_len, text_len]
        scale = 1 / (query.shape[-1] ** 0.5)
        attn_weights = (
            torch.matmul(query, key.transpose(-2, -1)) * scale
        )  # [batch, heads, seq_len, text_len]

        # Modulate attention weights (before softmax)
        if self.should_modulate():
            attn_weights = self.modulate_attention_weights(attn_weights)

        attn_probs = torch.softmax(
            attn_weights, dim=-1
        )  # [batch, heads, seq_len, text_len]

        # Apply attention to values: [batch, heads, seq_len, head_dim]
        hidden_states = torch.matmul(attn_probs, value)
        hidden_states = hidden_states.transpose(1, 2).flatten(
            2, 3
        )  # [batch, seq_len, dim]
        hidden_states = hidden_states.type_as(query)

        # Output projection: [batch, seq_len, dim]
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


def get_word_ids(text: str, word_phrase: str, tokenizer) -> np.ndarray:
    """Get token indices for a word or phrase in text"""
    full_tokens = tokenizer.encode(text)
    full_words = [tokenizer.decode([t]).strip() for t in full_tokens]

    phrase_tokens = tokenizer.encode(word_phrase)
    phrase_words = [tokenizer.decode([t]).strip() for t in phrase_tokens]

    if phrase_words and phrase_words[-1] == "</s>":
        phrase_words = phrase_words[:-1]

    out = []
    phrase_len = len(phrase_words)

    for i in range(len(full_words) - phrase_len + 1):
        match = True
        for j in range(phrase_len):
            full_word_clean = (
                full_words[i + j].replace("#", "").replace("</w>", "").strip()
            )
            phrase_word_clean = (
                phrase_words[j].replace("#", "").replace("</w>", "").strip()
            )

            if full_word_clean != phrase_word_clean:
                match = False
                break

        if match:
            out = list(range(i, i + phrase_len))
            break

    return np.array(out)
