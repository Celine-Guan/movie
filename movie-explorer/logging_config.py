import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config_loader import APP_DIR, get_logging_config

APP_LOGGER_NAME = "movie-explorer"
_CONFIGURED = False


def _log_dir() -> Path:
    log_cfg = get_logging_config()
    log_path = Path(log_cfg["directory"])
    if log_path.is_absolute():
        return log_path
    return APP_DIR / log_path


def setup_logging() -> logging.Logger:
    global _CONFIGURED

    log_cfg = get_logging_config()
    log_dir = _log_dir()
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger(APP_LOGGER_NAME)
    if _CONFIGURED:
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    for filename, level in (
        ("app.log", logging.INFO),
        ("errors.log", logging.ERROR),
        ("debug.log", logging.DEBUG),
    ):
        handler = RotatingFileHandler(
            log_dir / filename,
            maxBytes=log_cfg["max_bytes"],
            backupCount=log_cfg["backup_count"] + (2 if filename == "errors.log" else 0),
            encoding="utf-8",
        )
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    _CONFIGURED = True
    logger.info("Logging initialized. Log directory: %s", log_dir)
    return logger


def get_logger(module: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(f"{APP_LOGGER_NAME}.{module}")
