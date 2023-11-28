import logging
import os

import hydra
import torch
from info_nce import InfoNCE
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from datasets.utils import download_dataset, load_data, init_tasks
from utils import evaluate
from models import AttentionModel, FiLMedModel
from models.utils import log_network_summary, seed
from utils.specialization import *

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S", level=logging.INFO
)
log = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    log.info(f"Run Configuration: \n{OmegaConf.to_yaml(cfg)}")
    assert not (cfg.ac and cfg.strategy in ["all-specific", "all-specific"])

    # Fix random seed
    seed(cfg)

    # Check if GPUs available when cuda enabled
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        log.info("CUDA not available. Switching to CPU computing instead.")
        cfg.device = "cpu"

    # Download dataset if necessary
    if not os.path.exists(os.path.join(cfg.data_dir, cfg.dataset)):
        log.info("Download dataset from URL...")
        download_dataset(cfg)
        log.info("Successfully downloaded and extracted dataset from URL.")

    # Load train and test sets
    log.info(f"Loading datasets for training and evaluation...")
    init_tasks(cfg)
    init_dl, continual_dl, eval_dl, test_dl = load_data(cfg)
    log.info(f"Loaded training and evaluation sets.")

    # Initialize model
    if cfg.arch.name == "FiLM" and cfg.strategy in FILM_SPLIT.keys():
        model = FiLMedModel(cfg, FILM_SPLIT[cfg.strategy])
    elif cfg.arch.name == "Transformer" and cfg.strategy in ATTENTION_SPLIT.keys():
        model = AttentionModel(cfg, ATTENTION_SPLIT[cfg.strategy])
    else:
        raise Exception(
            f"The requested combination of {cfg.arch.name} model and {cfg.strategy} is not configured. "
            f"Check typos or implementation."
        )

    model.to(cfg.device)
    log.info(f"Initialized {cfg.arch.name} model instance.")

    # Log trainable network parameters
    log_network_summary(cfg, model, log)

    # Set up optimizer
    init_optim = torch.optim.AdamW(model.parameters(), lr=cfg.init_lr)
    cl_optim = torch.optim.AdamW(model.parameters(), lr=cfg.cl_lr)

    # Configure loss function
    # See https://github.com/RElbers/info-nce-pytorch
    loss_fn = InfoNCE(temperature=0.1)

    log.info("======== Phase I: Initialization ========")
    for e in range(cfg.init_epochs):
        for i, batch in tqdm(
            enumerate(init_dl), desc=f"Epoch {e + 1}", total=len(init_dl), position=0
        ):
            # Evaluate accuracy
            if i == 0 and e % 5 == 0:
                log_stats = evaluate(cfg, model, eval_dl)

            # Enable model training
            model.train()
            batch = [b.to(cfg.device) for b in batch]

            # Compute output and loss
            p1, z2, z3 = model(x1=batch[0], x2=batch[1], x3=batch[2], mission=batch[3])
            loss = loss_fn(p1, z2, z3)

            # Update parameters
            init_optim.zero_grad()
            loss.backward()
            init_optim.step()

            if i == 0 and e % 5 == 0:
                log.info(f"Initialization Epoch: {e + 1}")
                log.info(f"  Loss: {loss:.4f}")
                log.info(f"  Accuracy on init tasks: {log_stats['eval_init/acc']:.4f}")
                log.info(
                    f"  Accuracy on continual tasks: {log_stats['eval_continual/acc']:.4f}"
                )

    log.info("=========== Phase I completed ===========")

    # Freeze vision and language encoding as well as decoding projection params after init
    model.encoder_decoder.requires_grad_(False)

    log.info("====== Phase II: Continual training =====")
    for t in range(len(continual_dl)):
        for e in range(cfg.continual_epochs):
            # Check if adaptation or consolidation epoch
            if cfg.ac:
                if (e + 1) % cfg.adapt_freq == 0:
                    model.shared.requires_grad_(True)
                    model.specialized.requires_grad_(False)
                else:
                    model.shared.requires_grad_(False)
                    model.specialized.requires_grad_(True)

            for i, batch in tqdm(
                enumerate(continual_dl[t]),
                desc=f"Epoch {e + 1}",
                total=len(continual_dl[t]),
                position=0,
            ):
                # Evaluate accuracy
                if i == 0 and e % 5 == 0:
                    log_stats = evaluate(cfg, model, eval_dl)

                # Enable model training
                model.train()

                batch = [b.to(cfg.device) for b in batch]

                # Compute output and loss
                p1, z2, z3 = model(
                    x1=batch[0], x2=batch[1], x3=batch[2], mission=batch[3], task=t
                )
                loss = loss_fn(p1, z2, z3)

                # Update parameters
                cl_optim.zero_grad()
                loss.backward()
                cl_optim.step()

                if i == 0 and e % 5 == 0:
                    log.info(f"Task: {t} Epoch: {e}")
                    log.info(f"  Loss: {loss:.4f}")
                    log.info(
                        f"  Accuracy on init tasks: {log_stats['eval_init/acc']:.4f}"
                    )
                    log.info(
                        f"  Accuracy on continual tasks: {log_stats['eval_continual/acc']:.4f}"
                    )

    log.info("=========== Phase II completed ==========")

    log.info("=========== Final evaluation ============")

    log_stats = evaluate(cfg, model, test_dl, "test")
    log.info(f"Final test accuracy of {cfg.arch.name} model on {cfg.dataset}:")
    log.info(f"  Accuracy on init tasks: {log_stats['test_init/acc']:.4f}")
    log.info(f"  Accuracy on continual tasks: {log_stats['test_continual/acc']:.4f}")

    log.info("========== Evaluation completed =========")


if __name__ == "__main__":
    main()
