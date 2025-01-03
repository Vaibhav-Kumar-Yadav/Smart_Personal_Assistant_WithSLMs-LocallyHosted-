from typing import Generator
from datetime import datetime
import gc
import onnxruntime_genai as og
from logger import logger
from config import MODEL_CONFIG

class Phi3Handler:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.tokenizer_stream = None
        self.last_used = None

    def initialize(self) -> bool:
        try:
            logger.info(f"Initializing model from {self.model_path}")
            self.model = og.Model(self.model_path)
            self.tokenizer = og.Tokenizer(self.model)
            self.tokenizer_stream = self.tokenizer.create_stream()
            self.last_used = datetime.now()
            return True
        except FileNotFoundError:
            logger.error(f"Model file not found at {self.model_path}")
            return False
        except MemoryError:
            logger.error("Insufficient memory to load model")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during initialization: {e}")
            return False

    def cleanup(self):
        try:
            if self.tokenizer_stream:
                self.tokenizer_stream.close()
            if self.model:
                del self.model
            if self.tokenizer:
                del self.tokenizer
            self.model = None
            self.tokenizer = None
            self.tokenizer_stream = None
            gc.collect()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def generate_response(self, system_prompt: str, user_prompt: str, context: str = None, citations: str = None) -> Generator[str, None, None]:
        try:
            formatted_prompt = MODEL_CONFIG["chat_template"]["system"].format(
                system=system_prompt,
                context=context if context else "No specific context provided.",
                citations=citations if citations else "No citations available.",
                input=user_prompt
            )
            
            input_tokens = self.tokenizer.encode(formatted_prompt)
            params = og.GeneratorParams(self.model)
            params.input_ids = input_tokens
            params.set_search_options(**MODEL_CONFIG["search_options"])
            generator = og.Generator(self.model, params)

            full_response = ""
            while not generator.is_done():
                generator.compute_logits()
                generator.generate_next_token()
                new_token = generator.get_next_tokens()[0]
                decoded_token = self.tokenizer_stream.decode(new_token)
                full_response += decoded_token
                yield full_response.rstrip('\n')

        except Exception as e:
            logger.error(f"Error during generation: {e}")
            yield f"Error during generation: {str(e)}"
        finally:
            if 'generator' in locals():
                del generator
                gc.collect()