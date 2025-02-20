import logging

logger = logging.getLogger("KiwiTracker")


def _setup_logging_alembic(level="INFO"):
    l = logging.getLogger("alembic")
    l.setLevel(level)

    ch = logging.StreamHandler()
    ch.setLevel(level)

    #formatter = logging.Formatter("%(asctime)s - %(name)s - %(filename)s - %(levelname)s - %(message)s")
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
    ch.setFormatter(formatter)

    l.addHandler(ch)


def setup_logging(level="DEBUG"):
    # Setup Logging
    # create logger with 'my_application'
    logger.setLevel(level)

    # create file handler which logs even debug messages
    # print(os.getcwd())
    fh = logging.FileHandler("kiwitracker.log")
    fh.setLevel(level)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(level)

    # create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(filename)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    _setup_logging_alembic(level)
