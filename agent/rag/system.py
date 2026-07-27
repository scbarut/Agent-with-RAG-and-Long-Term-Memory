"""FAISS-backed RAG system with BLIP image-captioning support."""

import io
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import requests
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor


class RagSystem:
    """Manages a FAISS vector store enriched with BLIP image captions.

    On instantiation the class will attempt to load a previously saved index
    from ``index_dir``.  If none is found it falls back to building one from
    the CSV file at ``csv_file``.
    """

    DEFAULT_INDEX_DIR: str = "langchain_faiss_index"
    DEFAULT_EMBEDDING_MODEL: str = "google/embeddinggemma-300m"
    DEFAULT_IMAGE_MODEL: str = "Salesforce/blip-image-captioning-base"
    DEFAULT_CSV_FILE: str = "df_500.csv"

    def __init__(
        self,
        index_dir: str = DEFAULT_INDEX_DIR,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        image_model: str = DEFAULT_IMAGE_MODEL,
        csv_file: str = DEFAULT_CSV_FILE,
    ) -> None:
        self.index_dir = index_dir
        self.embedding_model_name = embedding_model
        self.image_model_name = image_model
        self.csv_file = csv_file
        self.next_doc_id: int = 0

        print("Initializing RagSystem...")
        self._initialize_models()
        self._setup_vectorstore()
        self._set_next_doc_id()
        print("RagSystem ready.")

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initialize_models(self) -> None:
        """Load the embedding model and the BLIP image-captioning model."""
        print("Loading HuggingFace embedding model...")
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"

        self.embeddings_model = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": True},
        )
        print(f"Embedding model loaded on {self.device.upper()}.")

        print("Loading BLIP image captioning model...")
        self.blip_processor: BlipProcessor = BlipProcessor.from_pretrained(self.image_model_name)
        self.blip_model: BlipForConditionalGeneration = BlipForConditionalGeneration.from_pretrained(
            self.image_model_name
        )
        print("BLIP model ready.")

    def _setup_vectorstore(self) -> None:
        """Load an existing vector store or create one from the CSV file."""
        self.vectorstore: Optional[FAISS] = self._load_vectorstore()
        if self.vectorstore is None:
            self.vectorstore = self._create_vectorstore()
            self._save_vectorstore()
        else:
            print(f"Vector store loaded. Document count: {self.vectorstore.index.ntotal}")
        print("Vector store ready.")

    def _set_next_doc_id(self) -> None:
        """Determine the next available document ID from the current store."""
        if self.vectorstore is None:
            self.next_doc_id = 0
            return
        try:
            all_docs = self.vectorstore.similarity_search("", k=self.vectorstore.index.ntotal)
            max_id = max(
                (int(doc.metadata.get("doc_id", 0)) for doc in all_docs),
                default=0,
            )
            self.next_doc_id = max_id + 1
        except Exception:
            self.next_doc_id = 0

    # ------------------------------------------------------------------
    # Vector store persistence
    # ------------------------------------------------------------------

    def _save_vectorstore(self) -> None:
        """Persist the FAISS index to ``self.index_dir``."""
        os.makedirs(self.index_dir, exist_ok=True)
        self.vectorstore.save_local(self.index_dir)
        print(f"Vector store saved to '{self.index_dir}'.")

    def _load_vectorstore(self) -> Optional[FAISS]:
        """Load a previously saved FAISS index, or return ``None``."""
        try:
            if os.path.exists(self.index_dir) and os.listdir(self.index_dir):
                vectorstore = FAISS.load_local(
                    self.index_dir,
                    self.embeddings_model,
                    allow_dangerous_deserialization=True,
                )
                print(f"Vector store loaded from '{self.index_dir}'.")
                return vectorstore
            print(f"No existing vector store found at '{self.index_dir}'.")
            return None
        except Exception as exc:
            print(f"Vector store load error: {exc}")
            return None

    def _create_vectorstore(self) -> FAISS:
        """Build a new FAISS vector store by processing the CSV file."""
        print("Creating new vector store from CSV...")
        df = pd.read_csv(self.csv_file)
        documents: list[Document] = []

        print(f"Processing {len(df)} rows...")
        for idx, row in df.iterrows():
            if idx % 10 == 0:
                print(f"Processed: {idx}/{len(df)}", end="\r")

            text_content = self._row_to_text(row)
            image_url = row.get("link")
            image_description = ""

            if pd.notna(image_url):
                image = self._download_image(image_url)
                if image:
                    image_description = self._generate_caption(image)

            combined_content = text_content
            if image_description:
                combined_content += f" | Image description: {image_description}"

            metadata = row.to_dict()
            metadata.update(
                {
                    "doc_id": str(idx),
                    "image_description": image_description,
                    "image_url": image_url,
                }
            )
            documents.append(Document(page_content=combined_content, metadata=metadata))

        print(f"\n{len(documents)} documents created.")
        print("Building FAISS index...")
        vectorstore = FAISS.from_documents(documents, self.embeddings_model)
        print(f"Vector store created. Document count: {vectorstore.index.ntotal}")
        return vectorstore

    # ------------------------------------------------------------------
    # Image helpers
    # ------------------------------------------------------------------

    def _generate_caption(self, image: Image.Image) -> str:
        """Generate a text caption for *image* using BLIP."""
        try:
            inputs = self.blip_processor(image, return_tensors="pt")
            out = self.blip_model.generate(**inputs, max_length=100, num_beams=5)
            return self.blip_processor.decode(out[0], skip_special_tokens=True)
        except Exception as exc:
            print(f"Image captioning error: {exc}")
            return "Image description unavailable"

    def _download_image(self, url: str, timeout: int = 10) -> Optional[Image.Image]:
        """Download and return an RGB image from *url*."""
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content)).convert("RGB")
        except Exception as exc:
            print(f"Image download error ({url}): {exc}")
            return None

    # ------------------------------------------------------------------
    # Text serialisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_text(row) -> str:
        """Serialise a DataFrame row to a pipe-separated ``key: value`` string."""
        return " | ".join(
            f"{col}: {val}" for col, val in row.items() if col != "link" and pd.notna(val)
        )

    @staticmethod
    def _dict_to_text(data: dict) -> str:
        """Serialise a dictionary to a pipe-separated ``key: value`` string."""
        return " | ".join(f"{k}: {v}" for k, v in data.items() if pd.notna(v))

    # ------------------------------------------------------------------
    # Input normalisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_text_input(text_data) -> list:
        """Coerce *text_data* to a list."""
        if isinstance(text_data, (str, dict)):
            return [text_data]
        if isinstance(text_data, list):
            return text_data
        return [str(text_data)]

    @staticmethod
    def _normalize_image_input(image_data, target_length: int) -> list:
        """Coerce *image_data* to a list of length *target_length*."""
        if image_data is None:
            return [None] * target_length
        if isinstance(image_data, Image.Image):
            return [image_data] * target_length
        if isinstance(image_data, list):
            padded = image_data[:target_length]
            padded += [None] * (target_length - len(padded))
            return padded
        return [None] * target_length

    @staticmethod
    def _normalize_metadata_input(metadata, target_length: int) -> list:
        """Coerce *metadata* to a list of dicts of length *target_length*."""
        if metadata is None:
            return [{}] * target_length
        if isinstance(metadata, dict):
            return [metadata] * target_length
        if isinstance(metadata, list):
            padded = metadata[:target_length]
            padded += [{}] * (target_length - len(padded))
            return padded
        return [{}] * target_length

    # ------------------------------------------------------------------
    # Public search API
    # ------------------------------------------------------------------

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """Return the top-*k* most similar documents for *query*."""
        if self.vectorstore is None:
            print("Vector store is not available.")
            return []
        try:
            return self.vectorstore.similarity_search(query, k=k)
        except Exception as exc:
            print(f"Search error: {exc}")
            return []

    def similarity_search_with_score(self, query: str, k: int = 3) -> List[tuple]:
        """Return the top-*k* documents together with their similarity scores."""
        if self.vectorstore is None:
            print("Vector store is not available.")
            return []
        try:
            return self.vectorstore.similarity_search_with_score(query, k=k)
        except Exception as exc:
            print(f"Search error: {exc}")
            return []

    def vector_search_text_img(
        self, query_text: str, image: Image.Image, k: int = 3
    ) -> List[Document]:
        """Search using a fused embedding of *query_text* and *image* caption.

        The text and image embeddings are averaged (simple fusion) before
        querying the vector store.
        """
        text_emb = np.array(self.embeddings_model.embed_query(query_text))
        image_caption = self._generate_caption(image)
        img_emb = np.array(self.embeddings_model.embed_query(image_caption))
        combined_emb = ((text_emb + img_emb) / 2).tolist()
        return self.vectorstore.similarity_search_by_vector(combined_emb, k=k)

    def vector_search_img(self, image: Image.Image, k: int = 3) -> List[Document]:
        """Search using only the caption embedding of *image*."""
        image_caption = self._generate_caption(image)
        img_emb = self.embeddings_model.embed_query(image_caption)
        return self.vectorstore.similarity_search_by_vector(img_emb, k=k)

    # ------------------------------------------------------------------
    # Public write API
    # ------------------------------------------------------------------

    def add_data(
        self,
        text_data: Union[str, Dict[str, Any], List[Dict[str, Any]]],
        image_data: Union[Image.Image, List[Optional[Image.Image]], None] = None,
        metadata: Union[Dict[str, Any], List[Dict[str, Any]], None] = None,
    ) -> bool:
        """Add one or more items to the vector store.

        Args:
            text_data: A string, dict, or list of dicts describing the data.
            image_data: An optional PIL Image (or list thereof) to caption and
                        associate with the text.
            metadata:   Optional extra metadata to attach to the stored documents.

        Returns:
            ``True`` on success, ``False`` on failure.
        """
        if self.vectorstore is None:
            print("Vector store is not available.")
            return False

        try:
            text_list = self._normalize_text_input(text_data)
            image_list = self._normalize_image_input(image_data, len(text_list))
            metadata_list = self._normalize_metadata_input(metadata, len(text_list))

            documents: list[Document] = []
            print(f"Processing {len(text_list)} item(s)...")

            for i, (text_item, image_item, meta_item) in enumerate(
                zip(text_list, image_list, metadata_list)
            ):
                print(f"Processing: {i + 1}/{len(text_list)}", end="\r")

                text_content = (
                    self._dict_to_text(text_item) if isinstance(text_item, dict) else str(text_item)
                )

                image_description = ""
                image_url = ""
                if image_item is not None:
                    if isinstance(image_item, Image.Image):
                        image_description = self._generate_caption(image_item)
                    elif isinstance(image_item, str):
                        image_url = image_item  # store URL as metadata only

                combined_content = text_content
                if image_description:
                    combined_content += f" | Image description: {image_description}"

                final_metadata: dict[str, Any] = {
                    "doc_id": str(self.next_doc_id + i),
                    "image_description": image_description,
                    "image_url": image_url,
                    "data_type": "text_only" if not image_description else "text_image",
                    "added_manually": True,
                }
                if meta_item:
                    final_metadata.update(meta_item)
                if isinstance(text_item, dict):
                    for key, value in text_item.items():
                        final_metadata.setdefault(key, value)

                documents.append(Document(page_content=combined_content, metadata=final_metadata))

            self.vectorstore.add_documents(documents)
            self.next_doc_id += len(documents)
            self._save_vectorstore()

            print(f"\n{len(documents)} item(s) added successfully.")
            return True

        except Exception as exc:
            print(f"Data add error: {exc}")
            return False

    def add_documents(self, documents: List[Document]) -> bool:
        """Add pre-built :class:`Document` objects to the vector store.

        Kept for backward compatibility with callers that construct documents
        themselves.
        """
        if self.vectorstore is None:
            print("Vector store is not available.")
            return False
        try:
            self.vectorstore.add_documents(documents)
            self._save_vectorstore()
            print(f"{len(documents)} document(s) added.")
            return True
        except Exception as exc:
            print(f"Document add error: {exc}")
            return False

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_info(self) -> dict:
        """Return metadata about the current vector store state."""
        if self.vectorstore is None:
            return {"error": "Vector store is not available"}
        return {
            "document_count": self.vectorstore.index.ntotal,
            "next_doc_id": self.next_doc_id,
            "index_dir": self.index_dir,
            "embedding_model": self.embedding_model_name,
            "image_model": self.image_model_name,
        }

    def get_vectorstore_info(self) -> dict:
        """Alias for :meth:`get_info`; kept for backward compatibility."""
        return self.get_info()
