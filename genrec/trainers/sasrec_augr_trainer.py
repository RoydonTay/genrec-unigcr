"""
SASRec AuGR Trainer - Self-Attentive Sequential Recommendation with CTR head.
"""
import os
import gin
import torch
import wandb

from genrec.models.sasrec_augr import SASRecWithRank
from genrec.modules.utils import parse_config, setup_logger
from genrec.data.amazon_sasrec import AmazonSASRecDataset, sasrec_collate_fn, sasrec_eval_collate_fn
from genrec.trainers.trainer_utils import (
    setup_accelerator, setup_wandb, save_checkpoint, get_parameter_count, log_training_info
)
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm


def evaluate(model, dataloader, accelerator, top_ks=[1, 5, 10]):
    """Evaluate model with Recall@K and NDCG@K."""
    model.eval()
    device = accelerator.device

    metrics = {f'Recall@{k}': 0.0 for k in top_ks}
    metrics.update({f'NDCG@{k}': 0.0 for k in top_ks})
    total = 0

    with torch.no_grad():
        for data in tqdm(dataloader, desc="Evaluating", disable=not accelerator.is_main_process):
            input_ids = data['input_ids'].to(device)
            targets = data['targets'].to(device)
            B = input_ids.size(0)

            wrapped_model = accelerator.unwrap_model(model)
            if getattr(wrapped_model, "gen_as_aux_task", False):
                hidden = wrapped_model._encode(input_ids)
                logits = hidden @ wrapped_model.item_embedding.weight.T
            else:
                logits, _ = wrapped_model(input_ids)

            if logits.dim() == 3:
                last_logits = logits[:, -1, :]
            elif logits.dim() == 2:
                last_logits = logits
            else:
                raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")
            last_logits[:, 0] = float('-inf')

            max_k = max(top_ks)
            _, top_k_items = torch.topk(last_logits, max_k, dim=-1)

            for i in range(B):
                target = targets[i].item()
                preds = top_k_items[i].tolist()

                for k in top_ks:
                    if target in preds[:k]:
                        metrics[f'Recall@{k}'] += 1.0
                        rank = preds[:k].index(target) + 1
                        metrics[f'NDCG@{k}'] += 1.0 / (torch.log2(torch.tensor(rank + 1.0)).item())

            total += B

    def gather(v):
        t = torch.tensor([v], device=device, dtype=torch.float32)
        return accelerator.reduce(t, reduction="sum").item()

    total = int(gather(total))
    for k in metrics:
        metrics[k] = gather(metrics[k]) / total if total > 0 else 0.0

    return metrics


@gin.configurable
def train(
    epochs=200, batch_size=128, learning_rate=1e-3, weight_decay=0.0,
    max_seq_len=50, embed_dim=64, num_heads=2, num_blocks=2, ffn_dim=256, dropout=0.2,
    loss_type="bce",
    dataset_folder="dataset/amazon", split="beauty",
    do_eval=True, eval_every_epoch=1, eval_batch_size=256,
    patience=50,
    save_dir_root="out/sasrec_augr/amazon/beauty", save_every_epoch=50,
    wandb_logging=False, wandb_project="sasrec_augr_training", wandb_log_interval=100,
    amp=True, mixed_precision_type="bf16", lambda_ctr=0.7, lambda_gen=0.3,
    ctr_hidden_units=[256, 128], ctr_mode="listnet", listnet_scale=20, gen_loss_decay=False,
    gen_as_aux_task=False, model_name=None, use_last_token_for_ctr=False, use_dot_product_logits=False,
):
    """Train SASRec AuGR model."""
    logger = setup_logger(save_dir_root, name="sasrec_augr")
    accelerator = setup_accelerator(amp=amp, mixed_precision_type=mixed_precision_type)
    device = accelerator.device

    if wandb_logging and accelerator.is_main_process:
        setup_wandb(
            run_name=model_name,
            project=wandb_project,
            config=locals(),
            step_metrics={"train/*": "global_step", "eval/*": "epoch"}
        )

    train_ds = AmazonSASRecDataset(root=dataset_folder, split=split, train_test_split="train", max_seq_len=max_seq_len)
    valid_ds = AmazonSASRecDataset(root=dataset_folder, split=split, train_test_split="valid", max_seq_len=max_seq_len)
    test_ds = AmazonSASRecDataset(root=dataset_folder, split=split, train_test_split="test", max_seq_len=max_seq_len)

    num_items = train_ds.num_items
    logger.info(f"Num items: {num_items}, Train: {len(train_ds)}, Valid: {len(valid_ds)}, Test: {len(test_ds)}")

    collate_train = lambda x: sasrec_collate_fn(x, max_seq_len, num_items=num_items if loss_type == "bce" else 0)
    collate_eval = lambda x: sasrec_eval_collate_fn(x, max_seq_len)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_train)
    valid_dl = DataLoader(valid_ds, batch_size=eval_batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_eval)
    test_dl = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_eval)

    steps_per_epoch = (len(train_ds) + batch_size - 1) // batch_size
    max_steps = steps_per_epoch * epochs
    logger.info(f"Steps per epoch: {steps_per_epoch}, Total max steps: {max_steps}")

    model = SASRecWithRank(
        num_items=num_items,
        max_seq_len=max_seq_len,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_blocks=num_blocks,
        ffn_dim=ffn_dim,
        dropout=dropout,
        loss_type=loss_type,
        lambda_ctr=lambda_ctr,
        lambda_gen=lambda_gen,
        ctr_hidden_units=ctr_hidden_units,
        ctr_mode=ctr_mode,
        listnet_scale=listnet_scale,
        gen_loss_decay=gen_loss_decay,
        max_steps=max_steps,
        gen_as_aux_task=gen_as_aux_task,
        use_last_token_for_ctr=use_last_token_for_ctr,
        use_dot_product_logits=use_dot_product_logits,
    )

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.98))
    train_dl, valid_dl, test_dl = accelerator.prepare(train_dl, valid_dl, test_dl)
    model, optimizer = accelerator.prepare(model, optimizer)

    logger.info(
        f"Model params: {get_parameter_count(model):,}, Loss: {loss_type}, CTR mode: {ctr_mode}, "
        f"Aux task: {gen_as_aux_task}, Dot-product CTR: {use_dot_product_logits}"
    )

    global_step = 0
    best_recall = 0.0
    wait = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_dl, desc=f"Epoch {epoch}", disable=not accelerator.is_main_process)
        for data in pbar:
            input_ids = data['input_ids']
            targets = data['targets']
            negatives = data.get('negatives', None)

            if loss_type == "bce":
                _, loss = model(input_ids, targets, negatives)
            else:
                _, loss = model(input_ids, targets)

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            global_step += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if wandb_logging and accelerator.is_main_process and global_step % wandb_log_interval == 0:
                wandb.log({"global_step": global_step, "train/loss": loss.item()})

        avg_loss = epoch_loss / len(train_dl)
        logger.info(f"Epoch {epoch} - loss: {avg_loss:.4f}")

        if do_eval and (epoch + 1) % eval_every_epoch == 0:
            metrics = evaluate(model, valid_dl, accelerator)
            test_metrics_epoch = evaluate(model, test_dl, accelerator)
            if accelerator.is_main_process:
                logger.info(f"Epoch {epoch} - Valid: " + ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()]))
                logger.info(f"Epoch {epoch} - Test:  " + ", ".join([f"{k}={v:.4f}" for k, v in test_metrics_epoch.items()]))
                if wandb_logging:
                    wandb.log({"epoch": epoch, **{f"eval/{k}": v for k, v in metrics.items()}, **{f"test_epoch/{k}": v for k, v in test_metrics_epoch.items()}})

                if metrics['Recall@10'] > best_recall:
                    best_recall = metrics['Recall@10']
                    save_path = os.path.join(save_dir_root, f"{model_name}_best_model.pt" if model_name else "best_model.pt")
                    torch.save(accelerator.unwrap_model(model).state_dict(), save_path)
                    logger.info(f"New best Recall@10: {best_recall:.4f}, saved to {save_path}")
                    wait = 0
                else:
                    wait += 1
                    logger.info(f"No improvement for {wait}/{patience} epochs")
                    if wait >= patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break

            model.train()

        if accelerator.is_main_process and (epoch + 1) % save_every_epoch == 0:
            save_path = os.path.join(save_dir_root, f"checkpoint_epoch_{epoch}.pt")
            torch.save(accelerator.unwrap_model(model).state_dict(), save_path)
            logger.info(f"Saved checkpoint to {save_path}")

    if accelerator.is_main_process:
        best_path = os.path.join(save_dir_root, f"{model_name}_best_model.pt" if model_name else "best_model.pt")
        if os.path.exists(best_path):
            accelerator.unwrap_model(model).load_state_dict(torch.load(best_path))

    test_metrics = evaluate(model, test_dl, accelerator)
    if accelerator.is_main_process:
        logger.info(f"Test Results: " + ", ".join([f"{k}={v:.4f}" for k, v in test_metrics.items()]))
        if wandb_logging:
            wandb.log({f"test/{k}": v for k, v in test_metrics.items()})
            wandb.finish()

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    parse_config()
    train()