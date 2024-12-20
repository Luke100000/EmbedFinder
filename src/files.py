import base64
import io
import queue
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Iterable, Optional, List

import chromadb
import numpy as np
import pathspec
from PIL import Image
from chromadb.errors import InvalidCollectionException
from diskcache import Cache

from embedding import Embedder, ImageEmbedder

DEFAULT_IMAGE_PATTERNS = (
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.bmp",
    "*.gif",
    "*.tiff",
    "*.webp",
    "*.ppm",
    "*.pgm",
    "*.pbm",
    "*.pnm",
    "*.ico",
    "*.dds",
    "*.tga",
    "*.heic",
    "*.heif",
)

DEFAULT_SOUND_PATTERNS = (
    "*.ogg",
    "*.mp3",
    "*.wav",
    "*.flac",
    "*.m4a",
    "*.aac",
    "*.wma",
    "*.aiff",
    "*.au",
)


@dataclass
class File:
    path: str
    size: int = 0
    modified: float = 0
    thumbnail: str = ""


def encode_thumbnail(image: Image.Image) -> str:
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    return base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")


class FileManager:
    def __init__(
        self,
        patterns: Iterable[str] = DEFAULT_IMAGE_PATTERNS,
        embedder: Embedder = ImageEmbedder(),
    ) -> None:
        Path("cache").mkdir(exist_ok=True)

        self.spec = pathspec.PathSpec.from_lines("gitwildmatch", patterns)

        self.chroma_client = chromadb.Client()
        self.chroma_collection = self.chroma_client.get_or_create_collection("files")

        self.embedder = embedder
        self.cache = Cache(".cache")

        self.queue = Queue()
        self.worker = Thread(target=self._embedder, args=(), daemon=True)
        self.worker.start()

        self.scan_queue = Queue()
        self.scanner = Thread(target=self._scanner, args=(), daemon=True)
        self.scanner.start()
        self.break_scan = False

    def scan(self, root: Optional[PathLike]) -> None:
        self.break_scan = True
        with self.queue.mutex:
            self.queue.queue.clear()
        with self.scan_queue.mutex:
            self.scan_queue.queue.clear()

        self.scan_queue.put(root)

    def _scanner(self) -> None:
        while True:
            root = self.scan_queue.get(block=True)
            self.break_scan = False

            self.chroma_client.delete_collection("files")
            self.chroma_collection = self.chroma_client.create_collection("files")

            for path in Path(root).rglob("*"):
                if self.break_scan:
                    break

                p = path.absolute().as_posix()
                if self.spec.match_file(p.lower()):
                    print("Scanning", p)
                    stats = path.stat()
                    data: dict = self.cache.get(p)
                    if data is None or data["modified"] != stats.st_mtime:
                        self.queue.put(
                            {
                                "path": path,
                                "size": stats.st_size,
                                "modified": stats.st_mtime,
                            }
                        )
                    else:
                        self.add(p, data)

    def _fetch_batch(self, batch_size: int) -> List[dict]:
        data = [self.queue.get(block=True)]
        for _ in range(batch_size - 1):
            try:
                data.append(self.queue.get(block=False))
            except queue.Empty:
                break
        return data

    def _embedder(self) -> None:
        while True:
            draw_datas = self._fetch_batch(self.embedder.get_batch_size())

            # Load and resize images
            datas = []
            metas = []
            for data in draw_datas:
                try:
                    datas.append(self.embedder.load_data(data["path"]))
                    metas.append(data)
                except Exception as e:
                    print("Error loading", data["path"], e)

            # Embed images
            embeds = (
                self.embedder.embed_data(
                    np.stack([np.array(image) for image in datas], axis=0)
                )
                if datas
                else []
            )

            # Encode thumbnails
            thumbnails = [self.embedder.get_thumbnail(data) for data in datas]

            for embed, thumbnail, data in zip(embeds, thumbnails, metas):
                p = data["path"].absolute().as_posix()
                del data["path"]

                data["embedding"] = embed
                data["thumbnail"] = encode_thumbnail(thumbnail)

                self.cache.set(p, data)
                self.add(p, data)
                print("Recalculated", p, embeds.shape)

    def add(self, p: str, data: dict) -> None:
        embedding = data["embedding"]
        del data["embedding"]

        try:
            self.chroma_collection.upsert(
                ids=p,
                metadatas=data,
                embeddings=embedding,
            )
            print("Added", p)
        except InvalidCollectionException:
            pass

    def search(self, query: str) -> List[File]:
        if self.embedder is None:
            return [File("Not ready yet...")]

        embedding = self.embedder.embed_query(query)
        try:
            result = self.chroma_collection.query(
                query_embeddings=embedding, n_results=6
            )
        except InvalidCollectionException:
            return [File("Not ready yet...")]

        files = []
        if not self.queue.empty():
            files.append(File(f"Loading {self.queue.qsize()} files..."))

        assert result["metadatas"]
        for path, meta in zip(result["ids"][0], result["metadatas"][0]):
            files.append(
                File(
                    path,
                    meta["size"],  # pyright: ignore [reportArgumentType]
                    meta["modified"],  # pyright: ignore [reportArgumentType]
                    meta["thumbnail"],  # pyright: ignore [reportArgumentType]
                )
            )
        return files
