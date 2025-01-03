import logging

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename='chatbot.log'
    )
    return logging.getLogger(__name__)

logger = setup_logger()