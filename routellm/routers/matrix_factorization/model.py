import torch
from huggingface_hub import PyTorchModelHubMixin
import requests
import logging
from sentence_transformers import SentenceTransformer
import numpy as np
from dotenv import load_dotenv
import os
import aiohttp
import asyncio
from openai import OpenAI
# Load environment variables from the .env file
load_dotenv()   

# Access the environment variable
openai_api_key = os.getenv('OPENAI_API_KEY')
llm_queue_gateway_base_path = os.getenv('LLM_QUEUE_GATEWAY_BASE_PATH')
use_openai_embedding  = os.getenv('USE_OPENAI_EMBEDDING', 'True').lower() == 'true'
use_ollama  = os.getenv('USE_OLLAMA_FOR_EMBEDDING', 'True').lower() == 'true'
embedding_model = os.getenv('EMBEDDING_MODEL', 'mxbai-embed-large')


openai_client = OpenAI(api_key=openai_api_key)

MODEL_IDS = {
    "RWKV-4-Raven-14B": 0,
    "alpaca-13b": 1,
    "chatglm-6b": 2,
    "chatglm2-6b": 3,
    "chatglm3-6b": 4,
    "claude-1": 5,
    "claude-2.0": 6,
    "claude-2.1": 7,
    "claude-instant-1": 8,
    "codellama-34b-instruct": 9,
    "deepseek-llm-67b-chat": 10,
    "dolly-v2-12b": 11,
    "dolphin-2.2.1-mistral-7b": 12,
    "falcon-180b-chat": 13,
    "fastchat-t5-3b": 14,
    "gemini-pro": 15,
    "gemini-pro-dev-api": 16,
    "gpt-3.5-turbo-0125": 17,
    "gpt-3.5-turbo-0314": 18,
    "gpt-3.5-turbo-0613": 19,
    "gpt-3.5-turbo-1106": 20,
    "gpt-4-0125-preview": 21,
    "gpt-4-0314": 22,
    "gpt-4-0613": 23,
    "gpt-4-1106-preview": 24,
    "gpt4all-13b-snoozy": 25,
    "guanaco-33b": 26,
    "koala-13b": 27,
    "llama-13b": 28,
    "llama-2-13b-chat": 29,
    "llama-2-70b-chat": 30,
    "llama-2-7b-chat": 31,
    "llama2-70b-steerlm-chat": 32,
    "mistral-7b-instruct": 33,
    "mistral-7b-instruct-v0.2": 34,
    "mistral-medium": 35,
    "mixtral-8x7b-instruct-v0.1": 36,
    "mpt-30b-chat": 37,
    "mpt-7b-chat": 38,
    "nous-hermes-2-mixtral-8x7b-dpo": 39,
    "oasst-pythia-12b": 40,
    "openchat-3.5": 41,
    "openchat-3.5-0106": 42,
    "openhermes-2.5-mistral-7b": 43,
    "palm-2": 44,
    "pplx-70b-online": 45,
    "pplx-7b-online": 46,
    "qwen-14b-chat": 47,
    "qwen1.5-4b-chat": 48,
    "qwen1.5-72b-chat": 49,
    "qwen1.5-7b-chat": 50,
    "solar-10.7b-instruct-v1.0": 51,
    "stablelm-tuned-alpha-7b": 52,
    "starling-lm-7b-alpha": 53,
    "stripedhyena-nous-7b": 54,
    "tulu-2-dpo-70b": 55,
    "vicuna-13b": 56,
    "vicuna-33b": 57,
    "vicuna-7b": 58,
    "wizardlm-13b": 59,
    "wizardlm-70b": 60,
    "yi-34b-chat": 61,
    "zephyr-7b-alpha": 62,
    "zephyr-7b-beta": 63,
}

class EmbeddingTransformer:
    def __init__(self, input_dim=384, output_dim=1536):
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x):
        # Convert to numpy if input is a PyTorch tensor
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()

        # Ensure the input is of the correct shape
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, but got {x.shape[-1]}")
        
        # Zero padding to expand to output_dim dimensions
        padded_vector = np.pad(x, (0, self.output_dim - self.input_dim), mode='constant')

        # Convert the padded numpy array back to torch tensor
        return torch.tensor(padded_vector, dtype=torch.float32)

class MFModel(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        dim,
        num_models,
        text_dim,
        num_classes,
        use_proj,
    ):
        super().__init__()
        self._name = "TextMF"
        self.use_proj = use_proj
        self.P = torch.nn.Embedding(num_models, dim)

        self.embedding_model = "text-embedding-3-small"

        if self.use_proj:
            self.text_proj = torch.nn.Sequential(
                torch.nn.Linear(text_dim, dim, bias=False)
            )
        else:
            assert (
                text_dim == dim
            ), f"text_dim {text_dim} must be equal to dim {dim} if not using projection"

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(dim, num_classes, bias=False)
        )

    def get_device(self):
        return self.P.weight.device

    def forward(self, model_id, prompt):
        try:
            logging.info('Starting forward pass')
            logging.info('model_id:')
            logging.info(model_id)
            
            # Move model_id to tensor and device
            device = self.get_device()
            model_id_tensor = torch.tensor(model_id, dtype=torch.long).to(device)

            # Get model embedding and normalize
            model_embed = torch.nn.functional.normalize(self.P(model_id_tensor), p=2, dim=1)

            logging.info('Prompt received: %s', prompt)
            # Generate prompt embedding
             # Run the async function synchronously
            
            prompt_embed = self.generate_embed(prompt)
            logging.info('Prompt embedding generated')

            # Convert prompt embedding to tensor and move to device
            prompt_embed_tensor = torch.tensor(prompt_embed, dtype=torch.float32).to(device)
            logging.info('Prompt embedding size: %s', prompt_embed_tensor.size(0))

            # Transform embedding
            # transformer = EmbeddingTransformer(input_dim=prompt_embed_tensor.size(0))
            # prompt_embed_transformed = transformer.forward(prompt_embed_tensor)
            # logging.info('Transformed prompt embedding size: %s', prompt_embed_transformed.size(0))

            # Project and classify
            # prompt_embed_projected = self.text_proj(prompt_embed_transformed)
            prompt_embed_projected = self.text_proj(prompt_embed_tensor)
            result = self.classifier(model_embed * prompt_embed_projected).squeeze()
            logging.info('Forward pass completed successfully')

            return result
        except requests.RequestException as e:
            logging.error('HTTP request failed: %s', e)
            raise RuntimeError('Failed to fetch embeddings from Ollama API') from e
        except Exception as e:
            logging.error('Error in forward pass: %s', e)
            raise RuntimeError('An error occurred during the forward pass') from e

    def generate_embed(self, prompt):
        try:
            if use_openai_embedding:
                logging.info('Generating embedding using OpenAI API')
                response = openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=prompt
                )
                embedding = response.data[0].embedding
                logging.info('Embedding received from OpenAI API')
                return embedding
            elif use_ollama:
                logging.info('Generating embedding using Ollama API')
                response = requests.post(
                    f'{llm_queue_gateway_base_path}/api/embeddings',
                    json={"model": embedding_model, "prompt": prompt}
                )
                response.raise_for_status()
                embedding = response.json()["embedding"]
                logging.info('Embedding received from Ollama API')
                return embedding
            else:
                logging.info('Generating embedding using SentenceTransformer')
                model = SentenceTransformer("avsolatorio/GIST-all-MiniLM-L6-v2")
                embeddings = model.encode(prompt, convert_to_tensor=False)
                logging.info('Embedding generated using SentenceTransformer')
                return embeddings
        except requests.RequestException as e:
            logging.error('Failed to fetch embedding from Ollama API: %s', e)
            raise
        except Exception as e:
            logging.error('Error generating embedding: %s', e)
            raise

    
    @torch.no_grad()
    def pred_win_rate(self, model_a, model_b, prompt):
        logits = self.forward([model_a, model_b], prompt)
        winrate = torch.sigmoid(logits[0] - logits[1]).item()
        return winrate

    def load(self, path):
        self.load_state_dict(torch.load(path))
