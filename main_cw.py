import random
import torch
import numpy as np
from config import get_config
from utils.io_utils import get_or_create_logger
from runner import CrossWOZRunner

logger = get_or_create_logger(__name__)

def main():
    cfg = get_config()

    # cuda setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = min(torch.cuda.device_count(), cfg.num_gpus)

    setattr(cfg, "device", device)
    setattr(cfg, "num_gpus", num_gpus)
    logger.info("Device: %s (the number of GPUs: %d)", str(device), num_gpus)

    if cfg.seed > 0:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

        logger.info("Set random seed to %d", cfg.seed)

    runner = CrossWOZRunner(cfg)

    if cfg.run_type == 'train':
        runner.train()
    else:
        runner.predict()

if __name__ == '__main__':
    main()