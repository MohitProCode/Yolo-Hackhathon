import logging
from pathlib import Path


def _has_handler(logger: logging.Logger, handler_type: type, predicate=None) -> bool:
    for handler in logger.handlers:
        if isinstance(handler, handler_type) and (predicate is None or predicate(handler)):
            return True
    return False


def create_logger(name: str = "seg", log_file: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    if not _has_handler(logger, logging.StreamHandler):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if log_file:
        log_path = Path(log_file).resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)

        def _match(handler: logging.Handler) -> bool:
            base = getattr(handler, "baseFilename", None)
            return base is not None and Path(base).resolve() == log_path

        if not _has_handler(logger, logging.FileHandler, _match):
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
