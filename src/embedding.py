import numpy as np
from cachetools import cached
from transformers import ClapModel, ClapProcessor, CLIPModel, CLIPProcessor
from PIL import Image
import librosa


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


def resize_with_crop(image: Image.Image, target_size: int = 224):
    width, height = image.size
    size = min(width, height)

    # Center crop to a square
    left = (width - size) // 2
    top = (height - size) // 2
    right = left + size
    bottom = top + size
    image = image.crop((left, top, right, bottom))

    # Resize to target size
    return image.resize((target_size, target_size), Image.Resampling.LANCZOS)


class Embedder:
    def get_batch_size(self) -> int:
        return 1

    def load_data(self, path: str) -> np.ndarray:
        raise NotImplementedError()

    def get_thumbnail(self, data: np.ndarray) -> Image.Image:
        return Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8))

    def embed_data(self, data: np.ndarray) -> np.ndarray:
        return data

    def embed_query(self, query: str) -> np.ndarray:
        return np.asarray(query)


class AudioEmbedder(Embedder):
    def load_data(self, path: str) -> np.ndarray:
        audio, _ = librosa.load(path, sr=48000, mono=True)
        return audio

    def embed_data(self, data: np.ndarray) -> np.ndarray:
        model, processor = get_audio_model()
        inputs = processor(audios=data, return_tensors="pt", sampling_rate=48000)
        audio_embed = model.get_audio_features(**inputs)
        return audio_embed.detach().numpy()

    def embed_query(self, query: str) -> np.ndarray:
        model, processor = get_audio_model()
        inputs = processor(text=query, return_tensors="pt")
        text_embed = model.get_text_features(**inputs)
        return text_embed.detach().numpy()


class ImageEmbedder(Embedder):
    def get_batch_size(self) -> int:
        return 16

    def load_data(self, path: str) -> np.ndarray:
        return np.asarray(resize_with_crop(Image.open(path).convert("RGB"), 224))

    def get_thumbnail(self, data: np.ndarray) -> Image.Image:
        image = Image.fromarray(data)
        image.thumbnail((128, 128))
        return image

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
