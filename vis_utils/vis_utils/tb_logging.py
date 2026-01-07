from torch.utils.tensorboard import SummaryWriter

# -------------------------
# Global logging state
# -------------------------

LOG_INTERVAL = 1  # you can change this
TI_LOG_STEP = 0  # global training iteration counter
TENSORBOARD_LOGGER = None


def init(log_dir):
    """
    Initialize the TensorBoard writer.
    Call this ONCE at startup.
    """
    global TENSORBOARD_LOGGER
    if TENSORBOARD_LOGGER is None:
        TENSORBOARD_LOGGER = SummaryWriter(log_dir)


def close():
    global TENSORBOARD_LOGGER
    if TENSORBOARD_LOGGER is not None:
        TENSORBOARD_LOGGER.close()
        TENSORBOARD_LOGGER = None
