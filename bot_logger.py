"""Logging setup: console + rotating daily file under logs/."""
import logging
import logging.handlers
import os


def setup(level: int = logging.INFO):
    os.makedirs("logs", exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)-22s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(level)

    # Console
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # Daily rotating file (keeps 30 days)
    fh = logging.handlers.TimedRotatingFileHandler(
        "logs/bot.log", when="midnight", backupCount=30, encoding="utf-8"
    )
    fh.setFormatter(fmt)
    root.addHandler(fh)
