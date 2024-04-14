import logging

logger = logging.getLogger("KiwiTracker")


def setup_logging():
    # Setup Logging
    # create logger with 'my_application'
    logger.setLevel(logging.DEBUG)

    # create file handler which logs even debug messages
    # print(os.getcwd())
    fh = logging.FileHandler("kiwitracker.log")
    fh.setLevel(logging.DEBUG)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(filename)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
