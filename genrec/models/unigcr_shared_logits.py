"""
HSTU: Hierarchical Sequential Transduction Unit
From "Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations"
https://arxiv.org/abs/2402.17152

Key differences from standard Transformer:
1. SiLU activation instead of softmax normalization (captures preference intensity)
2. Update gate U for gating mechanism
3. Relative attention bias with both position and temporal components
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class UniGCR(nn.Module):
    """
    UniGCR model for sequential recommendation.

    Architecture:
        Input -> Item Embedding + (optional) Temporal Encoding
              -> [HSTU Layer × num_blocks]
              -> Prediction (dot product with item embeddings)
    """

    def __init__(
        self,
        num_items: int,
        max_seq_len: int = 50,
        embed_dim: int = 64,
        num_heads: int = 2,
        num_blocks: int = 2,
        dropout: float = 0.2,
        num_position_buckets: int = 32,
        num_time_buckets: int = 64,
        max_position_distance: int = 128,
        use_temporal_bias: bool = True,
        lambda_ctr: float = 0.7,
        lambda_gen: float = 0.3,
        ctr_hidden_units: Tuple[int] = (256, 128),
        ctr_mode: str = "listnet",  # "bce" or "listnet"
        listnet_scale: float = 20,
        gen_loss_decay: bool = False,
        max_steps: int = 10000,
        gen_as_aux_task: bool = False,
        use_last_token_for_ctr: bool = False,
        use_dot_product_logits: bool = False,
    ):
        """
        Args:
            num_items: Total number of items
            max_seq_len: Maximum sequence length
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_blocks: Number of HSTU layers
            dropout: Dropout rate
            num_position_buckets: Number of buckets for position bias
            num_time_buckets: Number of buckets for temporal bias
            max_position_distance: Max distance for position bucketing
            use_temporal_bias: Whether to use temporal attention bias
            lambda_ctr: Weight for ctr loss
            lambda_gen: Weight for generation loss
        """
        super().__init__()
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.use_temporal_bias = use_temporal_bias
        self.lambda_ctr = lambda_ctr
        self.lambda_gen = lambda_gen
        self.ctr_mode = ctr_mode.lower()
        self.listnet_scale = listnet_scale
        self.gen_loss_decay = gen_loss_decay
        self.max_steps = max_steps
        self.global_step = 0
        if self.ctr_mode not in {"bce", "listnet"}:
            raise ValueError(f"Unsupported ctr_mode={ctr_mode}. Use 'bce' or 'listnet'.")

        # Item embedding (0 is padding)
        self.item_embedding = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)

        # Embedding dropout
        self.emb_dropout = nn.Dropout(dropout)

        # HSTU layers
        self.layers = nn.ModuleList([
            HSTULayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                num_position_buckets=num_position_buckets,
                num_time_buckets=num_time_buckets,
                max_position_distance=max_position_distance,
                use_temporal_bias=use_temporal_bias,
            )
            for _ in range(num_blocks)
        ])

        # CTR towers follow the same MLP pattern as the vocab CTR example.
        self.ctr_bce_tower = nn.Sequential(
            self._build_mlp(embed_dim, ctr_hidden_units, dropout),
            nn.Linear(ctr_hidden_units[-1], num_items + 1),
        )
        self.ctr_listwise_tower = nn.Sequential(
            self._build_mlp(embed_dim, ctr_hidden_units, dropout),
            nn.Linear(ctr_hidden_units[-1], num_items + 1),
        )

        # Final layer norm
        self.final_norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)
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

    def forward(
        self,
        input_ids: torch.Tensor,  # [B, L]
        timestamps: Optional[torch.Tensor] = None,  # [B, L] unix timestamps
        targets: Optional[torch.Tensor] = None,  # [B, L] for training
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            input_ids: Item ID sequence [B, L], 0 is padding
            timestamps: Unix timestamps [B, L], optional for temporal bias
            targets: Target item IDs [B, L] for loss computation

        Returns:
            logits: Prediction logits [B, L, num_items+1]
            loss: Weighted combined loss if training labels provided
        """
        B, L = input_ids.shape
        device = input_ids.device

        # Causal mask
        causal_mask = torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()

        # Padding mask
        padding_mask = (input_ids == 0)

        # Item embedding
        x = self.item_embedding(input_ids)  # [B, L, D]
        x = self.emb_dropout(x)

        # Apply HSTU layers
        for layer in self.layers:
            x = layer(x, causal_mask, padding_mask, timestamps)

        x = self.final_norm(x)

        # Prediction via dot product with item embeddings
        logits = x @ self.item_embedding.weight.T  # [B, L, V]

        # Compute loss
        gen_loss = None
        if targets is not None:
            gen_loss = F.cross_entropy(
                logits.view(-1, self.num_items + 1),
                targets.view(-1),
                ignore_index=0
            )

        ctr_loss = None
        if targets is not None:
            relevance = self._expand_targets_to_relevance(targets)  # [B, L, V]
            flat_targets = targets.reshape(-1)  # [B*L]
            flat_logits = logits.reshape(-1, self.num_items + 1)  # [B*L, V]
            flat_relevance = relevance.reshape(-1, self.num_items + 1)  # [B*L, V]
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

        return logits, total_loss

    def forward_sampled_softmax(
        self,
        input_ids: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        num_negatives: int = 128,
        temperature: float = 0.05,
        l2_norm: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward with sampled softmax loss (aligned with Meta's original HSTU).

        Instead of computing logits over ALL items, samples a small set of
        negatives per batch for efficiency and different optimization landscape.
        """
        B, L = input_ids.shape
        device = input_ids.device

        causal_mask = torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()
        padding_mask = (input_ids == 0)

        x = self.item_embedding(input_ids)
        x = self.emb_dropout(x)

        for layer in self.layers:
            x = layer(x, causal_mask, padding_mask, timestamps)

        x = self.final_norm(x)  # [B, L, D]

        # Full logits for eval (no loss)
        logits = x @ self.item_embedding.weight.T  # [B, L, V]

        gen_loss = None
        if targets is not None:
            # Flatten to [B*L, D] and [B*L]
            x_flat = x.view(-1, self.embed_dim)  # [B*L, D]
            targets_flat = targets.view(-1)  # [B*L]

            # Filter out padding positions
            valid_mask = targets_flat != 0
            x_valid = x_flat[valid_mask]  # [N, D]
            targets_valid = targets_flat[valid_mask]  # [N]

            if x_valid.size(0) > 0:
                # Get positive embeddings
                pos_emb = self.item_embedding(targets_valid)  # [N, D]

                # Sample random negatives (shared across batch for efficiency)
                neg_ids = torch.randint(1, self.num_items + 1, (num_negatives,), device=device)
                neg_emb = self.item_embedding(neg_ids)  # [K, D]

                # L2 normalize if enabled (as in Meta's implementation)
                if l2_norm:
                    x_valid = F.normalize(x_valid, dim=-1)
                    pos_emb = F.normalize(pos_emb, dim=-1)
                    neg_emb = F.normalize(neg_emb, dim=-1)

                # Compute logits: [N, 1+K]
                pos_logits = (x_valid * pos_emb).sum(dim=-1, keepdim=True)  # [N, 1]
                neg_logits = x_valid @ neg_emb.T  # [N, K]
                all_logits = torch.cat([pos_logits, neg_logits], dim=-1) / temperature

                # Target is always index 0 (positive)
                loss_targets = torch.zeros(x_valid.size(0), device=device, dtype=torch.long)
                gen_loss = F.cross_entropy(all_logits, loss_targets)

        return logits, gen_loss

    @torch.no_grad()
    def predict(self, input_ids: torch.Tensor, timestamps: Optional[torch.Tensor] = None, top_k: int = 10) -> torch.Tensor:
        """Predict top-k items for next item."""
        logits, _ = self.forward(input_ids, timestamps)
        if logits.dim() == 3:
            last_logits = logits[:, -1, :]
        elif logits.dim() == 2:
            last_logits = logits
        else:
            raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")
        last_logits[:, 0] = float('-inf')  # Exclude padding
        _, top_k_items = torch.topk(last_logits, top_k, dim=-1)
        return top_k_items


class HSTULayer(nn.Module):
    """
    Single HSTU layer.

    Structure:
        1. Pointwise Projection: X -> SiLU(Linear(X)) -> split to U, V, Q, K
        2. Spatial Aggregation: SiLU(QK^T + RAB) @ V
        3. Pointwise Transformation: Norm(Attention) ⊙ U -> FFN
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        num_position_buckets: int,
        num_time_buckets: int,
        max_position_distance: int,
        use_temporal_bias: bool,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_temporal_bias = use_temporal_bias

        assert embed_dim % num_heads == 0

        # Pointwise projection: projects to 4 * embed_dim (for U, V, Q, K)
        self.projection = nn.Linear(embed_dim, 4 * embed_dim)

        # Relative attention bias (position-based, shared across heads)
        self.position_bias = RelativePositionBias(
            num_buckets=num_position_buckets,
            max_distance=max_position_distance,
            num_heads=num_heads,
        )

        # Temporal bias (optional)
        if use_temporal_bias:
            self.temporal_bias = TemporalBias(
                num_buckets=num_time_buckets,
                num_heads=num_heads,
            )

        # Layer norm for attention output
        self.attn_norm = nn.LayerNorm(embed_dim)

        # FFN (pointwise transformation)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

        # Final layer norm
        self.ffn_norm = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,  # [B, L, D]
        causal_mask: torch.Tensor,  # [L, L]
        padding_mask: torch.Tensor,  # [B, L]
        timestamps: Optional[torch.Tensor] = None,  # [B, L]
    ) -> torch.Tensor:
        B, L, D = x.shape
        residual = x

        # === Pointwise Projection ===
        # Project and apply SiLU, then split into U, V, Q, K
        projected = F.silu(self.projection(x))  # [B, L, 4D]
        U, V, Q, K = projected.chunk(4, dim=-1)  # Each [B, L, D]

        # Reshape for multi-head attention
        Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, d]
        K = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # === Spatial Aggregation ===
        # Compute attention scores (without softmax!)
        scores = Q @ K.transpose(-2, -1)  # [B, H, L, L]

        # Add relative position bias
        pos_bias = self.position_bias(L, x.device)  # [H, L, L]
        scores = scores + pos_bias.unsqueeze(0)

        # Add temporal bias if enabled and timestamps provided
        if self.use_temporal_bias and timestamps is not None:
            time_bias = self.temporal_bias(timestamps)  # [B, H, L, L]
            scores = scores + time_bias

        # Apply causal mask (set masked positions to large negative)
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), -1e9)

        # Apply padding mask
        scores = scores.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2), -1e9)

        # HSTU key: SiLU instead of softmax!
        # This allows capturing preference intensity
        attn_weights = F.silu(scores)

        # Apply attention to values
        attn_output = attn_weights @ V  # [B, H, L, d]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)  # [B, L, D]

        # === Pointwise Transformation ===
        # Normalize and gate with U
        attn_output = self.attn_norm(attn_output)
        attn_output = attn_output * U  # Element-wise gating

        # Residual connection
        x = residual + self.dropout(attn_output)

        # FFN with residual
        x = x + self.ffn(self.ffn_norm(x))

        return x


class RelativePositionBias(nn.Module):
    """
    Relative position bias using logarithmic bucketing (T5-style).

    Buckets relative positions into logarithmically spaced bins,
    allowing the model to generalize to longer sequences.
    """

    def __init__(self, num_buckets: int = 32, max_distance: int = 128, num_heads: int = 2):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.num_heads = num_heads

        # Learnable bias for each bucket and head
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

    def _relative_position_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        """
        Convert relative position to bucket index using logarithmic bucketing.

        For causal attention, we only care about positions where query >= key,
        so relative_position >= 0.
        """
        # We use half buckets for exact positions, half for log-spaced
        num_buckets = self.num_buckets
        max_distance = self.max_distance

        # Clamp to non-negative (causal)
        relative_position = torch.clamp(relative_position, min=0)

        # Half buckets for small distances (exact)
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # Log-spaced buckets for larger distances
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).long()

        relative_position_if_large = torch.clamp(relative_position_if_large, max=num_buckets - 1)

        bucket = torch.where(is_small, relative_position, relative_position_if_large)
        return bucket

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Compute relative position bias matrix.

        Returns:
            bias: [num_heads, seq_len, seq_len]
        """
        # Create position indices
        positions = torch.arange(seq_len, device=device)
        # relative_position[i, j] = i - j (query_pos - key_pos)
        relative_position = positions.unsqueeze(0) - positions.unsqueeze(1)  # [L, L]

        # Convert to buckets
        buckets = self._relative_position_bucket(relative_position)  # [L, L]

        # Look up bias values
        bias = self.relative_attention_bias(buckets)  # [L, L, H]
        bias = bias.permute(2, 0, 1)  # [H, L, L]

        return bias


class TemporalBias(nn.Module):
    """
    Temporal attention bias using logarithmic bucketing of time differences.

    Quantizes timestamp differences into log-spaced buckets,
    capturing both recent and long-term temporal patterns.
    """

    def __init__(self, num_buckets: int = 64, num_heads: int = 2):
        super().__init__()
        self.num_buckets = num_buckets
        self.num_heads = num_heads

        # Learnable bias for each bucket and head
        self.temporal_attention_bias = nn.Embedding(num_buckets, num_heads)

    def _temporal_bucket(self, time_diff: torch.Tensor) -> torch.Tensor:
        """
        Convert time difference to bucket index.

        Uses formula: bucket = floor(log(max(1, |diff|)) / log_base)
        where log_base ≈ 0.301 (log10(2)) as in the paper
        """
        # Take absolute value and ensure minimum of 1
        abs_diff = torch.clamp(torch.abs(time_diff), min=1).float()

        # Log bucketing (using natural log, scaled)
        # Paper uses: floor(log(max(1, |diff|)) / 0.301)
        # We use a similar approach but cap at num_buckets - 1
        buckets = (torch.log(abs_diff) / 0.693).long()  # 0.693 = ln(2)
        buckets = torch.clamp(buckets, min=0, max=self.num_buckets - 1)

        return buckets

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal bias matrix.

        Args:
            timestamps: [B, L] unix timestamps

        Returns:
            bias: [B, num_heads, L, L]
        """
        B, L = timestamps.shape

        # Compute pairwise time differences
        # time_diff[i, j] = timestamps[i] - timestamps[j]
        time_diff = timestamps.unsqueeze(2) - timestamps.unsqueeze(1)  # [B, L, L]

        # Convert to buckets
        buckets = self._temporal_bucket(time_diff)  # [B, L, L]

        # Look up bias values
        bias = self.temporal_attention_bias(buckets)  # [B, L, L, H]
        bias = bias.permute(0, 3, 1, 2)  # [B, H, L, L]

        return bias
