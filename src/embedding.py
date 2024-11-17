import numpy as np
from cachetools import cached
from transformers import ClapModel, ClapProcessor, CLIPModel, CLIPProcessor


@cached({})
def get_audio_model(model_name: str = "laion/larger_clap_general"):
    model = ClapModel.from_pretrained(model_name)
    processor: ClapProcessor = ClapProcessor.from_pretrained(model_name)
    return model, processor


@cached({})
def get_image_model(model_name: str = "openai/clip-vit-base-patch32"):
    model = CLIPModel.from_pretrained(model_name)
    processor: CLIPProcessor = CLIPProcessor.from_pretrained(model_name)
    return model, processor


class Embedder:
    def embed_data(self, data: np.ndarray) -> np.ndarray:
        return data

    def embed_query(self, query: str) -> np.ndarray:
        return np.asarray(query)


class AudioEmbedder(Embedder):
    def embed_data(self, data: np.ndarray) -> np.ndarray:
        model, processor = get_audio_model()
        inputs = processor(audios=data, return_tensors="pt")
        audio_embed = model.get_audio_features(**inputs)
        return audio_embed.detach().numpy()

    def embed_query(self, query: str) -> np.ndarray:
        model, processor = get_audio_model()
        inputs = processor(text=query, return_tensors="pt")
        text_embed = model.get_text_features(**inputs)
        return text_embed.detach().numpy()


class ImageEmbedder(Embedder):
    def embed_data(self, data: np.ndarray) -> np.ndarray:
        model, processor = get_image_model()
        inputs = processor(images=data, return_tensors="pt")
        image_embed = model.get_image_features(**inputs)
        return image_embed.detach().numpy()

    def embed_query(self, query: str) -> np.ndarray:
        model, processor = get_image_model()
        inputs = processor(text=query, return_tensors="pt")
        text_embed = model.get_text_features(**inputs)
        return text_embed.detach().numpy()
