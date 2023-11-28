import torch
import torch.nn.functional as F


def evaluate(cfg, model, dl, mode="eval"):
    # Disable model training
    model.eval()

    correct_per_task = torch.zeros(len(dl), device=cfg.device)
    total_per_task = torch.zeros_like(correct_per_task)
    correct_per_instr = torch.zeros(
        len(cfg.init["indices"]) + len(cfg.continual["indices"]), device=cfg.device
    )
    total_per_instr = torch.zeros_like(correct_per_instr)

    for t in range(len(dl)):
        for j, batch in enumerate(dl[t]):
            batch = [b.to(cfg.device) for b in batch]

            # Compute output and loss
            with torch.no_grad():
                base, pos, neg = model(
                    x1=batch[0],
                    x2=batch[1],
                    x3=batch[2],
                    mission=batch[3],
                    task=t - 1 if t > 0 else None,
                )
            preds = F.cosine_similarity(base, pos, dim=-1) > F.cosine_similarity(
                base, neg, dim=-1
            )

            # Count correct samples per task
            correct_per_task[t] += torch.sum(preds)
            total_per_task[t] += len(preds)

            # Count correct samples per instruction
            instr_counts = torch.bincount(batch[4], minlength=len(total_per_instr))
            correct_per_instr += torch.bincount(
                batch[4], weights=preds.float(), minlength=len(total_per_instr)
            )
            total_per_instr += instr_counts

    # Total accuracy
    init_correct = correct_per_instr[cfg.init["indices"]]
    init_total = total_per_instr[cfg.init["indices"]]
    continual_correct = correct_per_instr[cfg.continual["indices"]]
    continual_total = total_per_instr[cfg.continual["indices"]]
    log_stats = {
        f"{mode}_init/acc": torch.mean(init_correct / init_total),
        f"{mode}_continual/acc": torch.mean(continual_correct / continual_total),
    }

    return log_stats
