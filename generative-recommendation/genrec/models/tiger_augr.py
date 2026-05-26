import torch
from torch import nn
import os
import torch.nn.functional as F

from .tiger import Tiger, TigerOutput

class TigerWithRanking(Tiger):
    def __init__(
        self,
        rank_loss_weight: float = 0.7,
        gen_loss_weight: float = 0.3,
        use_direct_catalog_loss: bool = False,
        catalog_mlp_hidden_units=(256, 128),
        catalog_mlp_dropout: float = 0.1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.rank_loss_weight = rank_loss_weight
        self.gen_loss_weight = gen_loss_weight
        self.use_direct_catalog_loss = use_direct_catalog_loss
        self.catalog_mlp_hidden_units = tuple(catalog_mlp_hidden_units)
        self.catalog_mlp_dropout = catalog_mlp_dropout
        self._catalog_mlp = None
        self._catalog_mlp_input_dim = None
        self._catalog_mlp_output_dim = None

    def _build_catalog_mlp(self, input_dim: int, output_dim: int) -> nn.Sequential:
        layers = []
        in_dim = input_dim
        for unit in self.catalog_mlp_hidden_units:
            layers.append(nn.Linear(in_dim, unit))
            layers.append(nn.PReLU())
            layers.append(nn.Dropout(self.catalog_mlp_dropout))
            in_dim = unit
        layers.append(nn.Linear(in_dim, output_dim))
        return nn.Sequential(*layers)

    def forward(
        self,
        user_input_ids: torch.Tensor,
        item_input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        target_input_ids: torch.Tensor,
        target_token_type_ids: torch.Tensor,
        seq_mask: torch.Tensor,
        valid_item_ids: torch.Tensor = None,
        target_item_indices: torch.Tensor = None,
        num_catalog_items: int = None,
    ) -> TigerOutput:
        base_out = super().forward(
            user_input_ids=user_input_ids,
            item_input_ids=item_input_ids,
            token_type_ids=token_type_ids,
            target_input_ids=target_input_ids,
            target_token_type_ids=target_token_type_ids,
            seq_mask=seq_mask,
        )

        gen_loss = base_out.loss
        rank_loss = None
        if target_input_ids is not None:
            rank_logits = base_out.logits[:, :-1, :]
            if self.use_direct_catalog_loss:
                if target_item_indices is None:
                    raise ValueError("target_item_indices must be provided when use_direct_catalog_loss=True.")
                if num_catalog_items is None:
                    if valid_item_ids is not None:
                        num_catalog_items = valid_item_ids.size(1) if valid_item_ids.dim() == 3 else valid_item_ids.size(0)
                    else:
                        raise ValueError("num_catalog_items must be provided when use_direct_catalog_loss=True.")
                rank_loss = self.compute_direct_catalog_loss(
                    rank_logits,
                    target_item_indices,
                    num_catalog_items=num_catalog_items,
                )
            else:
                rank_loss = self.compute_sid_ranking_loss(
                    rank_logits,
                    target_input_ids,
                    valid_item_ids=valid_item_ids,
                )

        total_loss = None
        if gen_loss is not None and rank_loss is not None:
            total_loss = self.gen_loss_weight * gen_loss + self.rank_loss_weight * rank_loss
        elif gen_loss is not None:
            total_loss = self.gen_loss_weight * gen_loss
        elif rank_loss is not None:
            total_loss = self.rank_loss_weight * rank_loss

        return TigerOutput(logits=base_out.logits, loss=total_loss)


    def compute_direct_catalog_loss(
        self,
        logits: torch.Tensor,
        target_item_indices: torch.Tensor,
        num_catalog_items: int,
    ) -> torch.Tensor:
        """
        Direct catalog loss that projects flattened SID logits to catalog space.

        Args:
            logits: (B, S, V) raw model logits for S SID steps.
            target_item_indices: (B,) global item indices in the catalog.
            num_catalog_items: total catalog size.
        """
        if logits.dim() != 3:
            raise ValueError(f"Expected logits with shape (B, S, V), got {tuple(logits.shape)}")

        B, S, V = logits.shape
        flattened_logits = logits.reshape(B, -1)

        expected_in = S * V
        if (
            self._catalog_mlp is None
            or self._catalog_mlp_input_dim != expected_in
            or self._catalog_mlp_output_dim != num_catalog_items
        ):
            self._catalog_mlp = self._build_catalog_mlp(expected_in, num_catalog_items)
            self._catalog_mlp = self._catalog_mlp.to(device=logits.device, dtype=logits.dtype)
            self._catalog_mlp_input_dim = expected_in
            self._catalog_mlp_output_dim = num_catalog_items

        catalog_scores = self._catalog_mlp(flattened_logits)
        return F.cross_entropy(catalog_scores, target_item_indices)


    def compute_sid_ranking_loss(self, logits: torch.Tensor, target_item_ids: torch.Tensor, valid_item_ids: torch.Tensor = None) -> torch.Tensor:
        """
        SID likelihood ranking loss over the catalog.

        Args:
            logits: (B, S, V) raw model logits.
            target_item_ids: (B, S) ground-truth SIDs (per-step codebook tokens).
            valid_item_ids: (N, S) or (B, N, S) catalog SIDs.
        """
        if valid_item_ids is None:
            if not hasattr(self, "valid_item_ids") or self.valid_item_ids is None:
                raise ValueError("valid_item_ids must be provided or set on the model as `valid_item_ids`.")
            valid_item_ids = self.valid_item_ids

        if logits.dim() != 3:
            raise ValueError(f"Expected logits with shape (B, S, V), got {tuple(logits.shape)}")

        B, S, V = logits.shape
        NIE = self.num_item_embeddings

        log_probs = F.log_softmax(logits, dim=-1)

        # Catalog scoring: sum log-probabilities across SID positions for each catalog item.
        if valid_item_ids.dim() == 2:
            valid_item_ids = valid_item_ids.unsqueeze(0).expand(B, -1, -1)
        elif valid_item_ids.dim() != 3:
            raise ValueError(f"Expected valid_item_ids with shape (N, S) or (B, N, S), got {tuple(valid_item_ids.shape)}")

        if valid_item_ids.size(2) != S:
            raise ValueError(f"valid_item_ids has S={valid_item_ids.size(2)} but logits has S={S}.")

        valid_item_ids = valid_item_ids.to(logits.device)
        N = valid_item_ids.size(1)
        item_scores = torch.zeros(B, N, device=logits.device)

        for step in range(S):
            step_tokens = valid_item_ids[:, :, step]
            vocab_ids = step_tokens + step * NIE
            step_log_probs = torch.gather(log_probs[:, step, :], 1, vocab_ids)
            item_scores += step_log_probs


        # Map target SIDs to catalog indices using base-NIE encoding.
        def _encode_sid(ids: torch.Tensor) -> torch.Tensor:
            enc = torch.zeros(ids.size(0), device=ids.device, dtype=torch.long)
            for step in range(ids.size(1)):
                enc = enc * NIE + ids[:, step]
            return enc

        target_enc = _encode_sid(target_item_ids)
        if valid_item_ids.size(0) == 1:
            catalog_enc = _encode_sid(valid_item_ids[0])
            match = catalog_enc.unsqueeze(0) == target_enc.unsqueeze(1)
        else:
            catalog_enc = torch.stack([_encode_sid(valid_item_ids[b]) for b in range(B)], dim=0)
            match = catalog_enc == target_enc.unsqueeze(1)

        if not match.any(dim=1).all():
            raise ValueError("Some target_item_ids are not present in valid_item_ids.")

        target_indices = match.float().argmax(dim=1)

        # Cross-entropy over catalog items; uses item_scores as logits.
        loss = F.cross_entropy(item_scores, target_indices)
        return loss

  
