"""
SASRec: Self-Attentive Sequential Recommendation
https://arxiv.org/abs/1808.09781

A self-attention based sequential recommendation model that adaptively
considers consumed items for next-item prediction.

This implementation follows the official TensorFlow implementation:
https://github.com/kang205/SASRec
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class SASRecWithRank(nn.Module):
    """
    SASRec model implementation following the official TF version.

    Key details from official implementation:
    1. Embedding scaled by sqrt(d)
    2. Key masking based on embedding values (zero = padding)
    3. Query masking after softmax (zero out padding positions)
    4. Residual connection inside attention
    5. Mask applied after each block
    """

    def __init__(
        self,
        num_items: int,
        max_seq_len: int = 50,
        embed_dim: int = 64,
        num_heads: int = 2,
        num_blocks: int = 2,
        ffn_dim: int = 256,
        dropout: float = 0.2,
        loss_type: str = "bce",  # "bce" (original paper) or "ce" (cross-entropy)
        ctr_hidden_units: Tuple[int] = (256, 128),
        ctr_mode: str = "listnet",  # "bce" or "listnet"
        listnet_scale: float = 20,
        gen_loss_decay: bool = False,
        max_steps: int = 10000,
        gen_as_aux_task: bool = False,
        lambda_ctr: float = 0.7,
        lambda_gen: float = 0.3,
        use_last_token_for_ctr: bool = False,
        use_dot_product_logits: bool = False,
    ):
        super().__init__()
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.loss_type = loss_type
        self.ctr_mode = ctr_mode.lower()
        self.listnet_scale = listnet_scale
        self.gen_loss_decay = gen_loss_decay
        self.max_steps = max_steps
        self.global_step = 0
        self.gen_as_aux_task = gen_as_aux_task
        self.lambda_ctr = lambda_ctr
        self.lambda_gen = lambda_gen
        self.use_last_token_for_ctr = use_last_token_for_ctr
        self.use_dot_product_logits = use_dot_product_logits
        assert use_last_token_for_ctr & use_dot_product_logits == False, "use_last_token_for_ctr and use_dot_product_logits cannot both be True (ambiguous CTR input)"

        if self.ctr_mode not in {"bce", "listnet"}:
            raise ValueError(f"Unsupported ctr_mode={ctr_mode}. Use 'bce' or 'listnet'.")

        # Embeddings: item 0 is padding (will be zeroed)
        self.item_embedding = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)

        # Dropout for embeddings
        self.emb_dropout = nn.Dropout(dropout)

        # Self-attention blocks
        self.blocks = nn.ModuleList([
            SASRecBlock(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_blocks)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(embed_dim, eps=1e-8)

        if self.use_dot_product_logits:
            ctr_input_dim = num_items + 1
        else:
            ctr_input_dim = embed_dim

        self.ctr_bce_tower = nn.Sequential(
            self._build_mlp(ctr_input_dim, ctr_hidden_units, dropout),
            nn.Linear(ctr_hidden_units[-1], num_items + 1),
        )
        self.ctr_listwise_tower = nn.Sequential(
            self._build_mlp(ctr_input_dim, ctr_hidden_units, dropout),
            nn.Linear(ctr_hidden_units[-1], num_items + 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _build_mlp(self, input_dim, hidden_units, dropout):
        layers = []
        in_dim = input_dim
        for unit in hidden_units:
            layers.append(nn.Linear(in_dim, unit))
            layers.append(nn.PReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = unit
        return nn.Sequential(*layers)

    def _expand_targets_to_relevance(self, targets: torch.Tensor) -> torch.Tensor:
        """Expand next-item ids [B, L] to dense relevance matrix [B, L, V]."""
        relevance = torch.zeros(
            targets.size(0),
            targets.size(1),
            self.num_items + 1,
            device=targets.device,
            dtype=torch.float,
        )
        if (targets != 0).any():
            relevance.scatter_(2, targets.unsqueeze(-1), 1.0)
            relevance[..., 0] = 0.0
        return relevance

    def _listnet_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """ListNet loss with scaled softmax targets and epsilon-stabilized log."""
        eps = 1e-8

        p_label = F.softmax(labels * self.listnet_scale, dim=1)
        p_pred = F.softmax(logits * self.listnet_scale, dim=1)
        loss = -(p_label * torch.log(p_pred + eps)).sum(dim=1)

        if reduction == "mean":
            return loss.mean()
        if reduction == "sum":
            return loss.sum()
        if reduction == "none":
            return loss
        return loss.mean()

    def _calc_aux_loss_weight(self, global_step: int, max_steps: int) -> float:
        """Linear decay schedule for auxiliary generation loss weight."""
        if max_steps <= 0:
            return 1.0

        decay_start = int(0.1 * max_steps)
        decay_end = int(0.5 * max_steps)

        if global_step <= decay_start:
            return 1.0
        if global_step <= decay_end:
            denom = max(decay_end - decay_start, 1)
            return 1.0 - (global_step - decay_start) / denom
        return 0.0

    def _encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Encode input sequence into hidden representations [B, L, D]."""
        B, L = input_ids.shape
        device = input_ids.device

        mask = (input_ids != 0).unsqueeze(-1).float()
        x = self.item_embedding(input_ids)
        if self.loss_type != "bce":
            x = x * (self.embed_dim ** 0.5)

        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        x = x + self.position_embedding(positions)

        x = self.emb_dropout(x)
        x = x * mask

        for block in self.blocks:
            x = block(x, mask)
            x = x * mask

        x = self.final_norm(x)
        return x

    def forward(
        self,
        input_ids: torch.Tensor,  # [B, L]
        targets: Optional[torch.Tensor] = None,  # [B, L]
        negatives: Optional[torch.Tensor] = None,  # [B, L] for BCE loss
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            input_ids: Item IDs sequence [B, L], 0 is padding
            targets: Target item IDs [B, L] for computing loss
            negatives: Negative sample IDs [B, L], required when loss_type="bce"

        Returns:
            logits: Prediction logits [B, L, num_items+1]
            loss: Training loss if targets provided, else None
        """
        B, L = input_ids.shape
        device = input_ids.device
        hidden = self._encode(input_ids)  # [B, L, D]

        gen_loss = None
        if targets is not None:
            if self.loss_type == "bce":
                assert negatives is not None, "negatives required for BCE loss"
                pos_emb = self.item_embedding(targets)    # [B, L, D]
                neg_emb = self.item_embedding(negatives)  # [B, L, D]
                pos_logits = (hidden * pos_emb).sum(dim=-1)  # [B, L]
                neg_logits = (hidden * neg_emb).sum(dim=-1)  # [B, L]

                valid_mask = (targets != 0).float()
                pos_loss = F.binary_cross_entropy_with_logits(
                    pos_logits, torch.ones_like(pos_logits), reduction='none'
                )
                neg_loss = F.binary_cross_entropy_with_logits(
                    neg_logits, torch.zeros_like(neg_logits), reduction='none'
                )
                gen_loss = ((pos_loss + neg_loss) * valid_mask).sum() / valid_mask.sum()
            else:
                logits_all = hidden @ self.item_embedding.weight.T
                gen_loss = F.cross_entropy(
                    logits_all.view(-1, self.num_items + 1),
                    targets.view(-1),
                    ignore_index=0,
                )

        logits = hidden @ self.item_embedding.weight.T  # [B, L, num_items+1]

        ctr_logits = None
        if self.use_last_token_for_ctr:
            valid_input_mask = input_ids != 0
            last_token_idx = valid_input_mask.long().sum(dim=1).sub(1).clamp(min=0)
            batch_idx = torch.arange(B, device=device)
            x_last = hidden[batch_idx, last_token_idx]
            ctr_logits = self.ctr_bce_tower(x_last) if self.ctr_mode == "bce" else self.ctr_listwise_tower(x_last)
        elif self.use_dot_product_logits:
            ctr_logits = self.ctr_bce_tower(logits) if self.ctr_mode == "bce" else self.ctr_listwise_tower(logits)
        else:
            ctr_logits = self.ctr_bce_tower(hidden) if self.ctr_mode == "bce" else self.ctr_listwise_tower(hidden)

        ctr_loss = None
        if targets is not None:
            relevance = self._expand_targets_to_relevance(targets)
            if self.use_last_token_for_ctr:
                seq_targets = targets[batch_idx, last_token_idx]
                flat_logits = ctr_logits
                flat_relevance = relevance[batch_idx, last_token_idx, :]
                valid_mask = seq_targets != 0
            else:
                flat_targets = targets.reshape(-1)
                flat_logits = ctr_logits.reshape(-1, self.num_items + 1)
                flat_relevance = relevance.reshape(-1, self.num_items + 1)
                valid_mask = flat_targets != 0

            if self.ctr_mode == "bce":
                if valid_mask.any():
                    ctr_loss = F.binary_cross_entropy_with_logits(
                        flat_logits[valid_mask, 1:],
                        flat_relevance[valid_mask, 1:],
                    )
            elif self.ctr_mode == "listnet":
                if valid_mask.any():
                    ctr_loss = self._listnet_loss(
                        flat_logits[valid_mask, 1:],
                        flat_relevance[valid_mask, 1:],
                        reduction="mean",
                    )
            else:
                raise ValueError(f"Unsupported ctr_mode={self.ctr_mode}")

        total_loss = None
        gen_weight = self.lambda_gen
        if self.gen_loss_decay:
            gen_weight *= self._calc_aux_loss_weight(self.global_step, self.max_steps)

        if gen_loss is not None and ctr_loss is not None:
            total_loss = gen_weight * gen_loss + self.lambda_ctr * ctr_loss
        elif gen_loss is not None:
            total_loss = gen_weight * gen_loss
        elif ctr_loss is not None:
            total_loss = ctr_loss

        if self.training and targets is not None:
            self.global_step += 1

        if self.gen_as_aux_task:
            return ctr_logits, total_loss

        return logits, total_loss

    @torch.no_grad()
    def predict(self, input_ids: torch.Tensor, top_k: int = 10) -> torch.Tensor:
        """Predict top-k items for next item."""
        logits, _ = self.forward(input_ids)
        last_logits = logits[:, -1, :]
        last_logits[:, 0] = float('-inf')  # Exclude padding
        _, top_k_items = torch.topk(last_logits, top_k, dim=-1)
        return top_k_items


class SASRecBlock(nn.Module):
    """Single self-attention block for SASRec (following official impl)."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = PointWiseFeedForward(embed_dim, ffn_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-8)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-8)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, L, D]
            mask: Valid position mask [B, L, 1], True=valid, False=padding
        """
        # Self-attention: normalize queries only (as in official impl)
        # Residual is added inside attention
        x = self.attention(self.norm1(x), x, mask)

        # Feed-forward with residual (added inside ffn)
        x = self.ffn(self.norm2(x), x)

        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention following official SASRec implementation.

    Key differences from standard attention:
    1. Key masking based on embedding values (sum of abs)
    2. Query masking after softmax
    3. Use large negative number instead of -inf
    4. Residual connection inside this module
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,  # [B, L, D] - normalized
        key_value: torch.Tensor,  # [B, L, D] - not normalized (original x)
        mask: torch.Tensor,  # [B, L, 1]
    ) -> torch.Tensor:
        B, L, _ = query.shape

        # Project Q from normalized input, K/V from original input
        Q = self.q_proj(query)
        K = self.k_proj(key_value)
        V = self.v_proj(key_value)

        # Reshape for multi-head: [B, L, D] -> [B, H, L, D_h]
        Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores: [B, H, L, L]
        scores = (Q @ K.transpose(-2, -1)) * self.scale

        # Key masking: based on whether key positions are padding
        # In official impl: key_masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1))
        # Since padding embeddings are zero, we use the mask directly
        # mask: [B, L, 1] -> key_mask: [B, 1, 1, L]
        key_mask = mask.squeeze(-1).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]

        # Use large negative number instead of -inf (official uses -2^32+1)
        padding_value = -1e9
        scores = scores.masked_fill(key_mask == 0, padding_value)

        # Causal masking (future blinding)
        causal_mask = torch.triu(torch.ones(L, L, device=scores.device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), padding_value)

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)

        # Query masking: zero out attention weights for padding query positions
        # This is done AFTER softmax in official impl
        query_mask = mask.squeeze(-1).unsqueeze(1).unsqueeze(-1)  # [B, 1, L, 1]
        attn_weights = attn_weights * query_mask

        # Dropout
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = attn_weights @ V  # [B, H, L, D_h]
        out = out.transpose(1, 2).contiguous().view(B, L, self.embed_dim)

        # Residual connection (inside attention as in official impl)
        # Note: residual uses the normalized query, not original x
        out = out + query

        return out


class PointWiseFeedForward(nn.Module):
    """Point-wise feed-forward network with residual inside."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Normalized input [B, L, D]
            residual: Original input for residual connection [B, L, D]
        """
        out = self.fc2(self.dropout(F.relu(self.fc1(x))))
        out = self.dropout(out)
        return out + residual  # Residual connection
