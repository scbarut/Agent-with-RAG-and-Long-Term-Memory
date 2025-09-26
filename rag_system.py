import os
import io
import torch
import pandas as pd
import requests
from PIL import Image
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from transformers import BlipProcessor, BlipForConditionalGeneration
from typing import Union, List, Dict, Any

class RagSystem:
    def __init__(self, 
                 index_dir="langchain_faiss_index",
                 embedding_model="google/embeddinggemma-300m",
                 image_model="Salesforce/blip-image-captioning-base",
                 csv_file='df_500.csv'):
        
        self.INDEX_DIR = index_dir
        self.EMBEDDING_MODEL = embedding_model
        self.IMAGE_MODEL = image_model
        self.csv_file = csv_file
        self.next_doc_id = 0  # Yeni dökümanlar için ID sayacı
        
        print("✅ LangChain konfigürasyonu hazır!")
        
        # Initialize models
        self._initialize_models()
        
        # Load or create vectorstore
        self._setup_vectorstore()
        
        # Set next document ID
        self._set_next_doc_id()
        
        print("✅ RagSystem hazır!")
    
    def _initialize_models(self):
        """Model ve embeddings'leri başlatır"""
        print("🔄 HuggingFace embeddings modeli yükleniyor...")
        
        # Device configuration
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Embeddings model
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name=self.EMBEDDING_MODEL,
            model_kwargs={'device': self.device},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        print(f"✅ Embeddings modeli için kullanılan cihaz: {self.device.upper()}")
        
        print("🔄 BLIP görsel açıklama modeli yükleniyor...")
        
        # BLIP model for image captioning
        self.blip_processor = BlipProcessor.from_pretrained(self.IMAGE_MODEL)
        self.blip_model = BlipForConditionalGeneration.from_pretrained(self.IMAGE_MODEL)
        
        print("✅ BLIP modeli hazır!")
    
    def _generate_image_description(self, image):
        """Görsel için açıklama üretir"""
        try:
            inputs = self.blip_processor(image, return_tensors="pt")
            out = self.blip_model.generate(**inputs, max_length=100, num_beams=5)
            description = self.blip_processor.decode(out[0], skip_special_tokens=True)
            return description
        except Exception as e:
            print(f"❌ Görsel açıklama hatası: {e}")
            return "Görsel açıklama üretilemedi"
    
    def _download_image(self, url: str, timeout: int = 10):
        """URL'den görsel indirir"""
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content))
            return image.convert('RGB')
        except Exception as e:
            print(f"❌ Görsel indirme hatası {url}: {e}")
            return None
    
    def _create_text_content(self, row):
        """DataFrame satırından metin içeriği oluşturur"""
        text_parts = []
        for col, val in row.items():
            if col != 'link' and pd.notna(val):
                text_parts.append(f"{col}: {val}")
        return " | ".join(text_parts)
    
    def _save_vectorstore(self):
        """LangChain FAISS vectorstore'u kaydeder"""
        os.makedirs(self.INDEX_DIR, exist_ok=True)
        self.vectorstore.save_local(self.INDEX_DIR)
        print(f"✅ VectorStore {self.INDEX_DIR} klasörüne kaydedildi")
    
    def _load_vectorstore(self):
        """LangChain FAISS vectorstore'u yükler"""
        try:
            if os.path.exists(self.INDEX_DIR) and os.listdir(self.INDEX_DIR):
                vectorstore = FAISS.load_local(
                    self.INDEX_DIR, 
                    self.embeddings_model, 
                    allow_dangerous_deserialization=True
                )
                print(f"✅ VectorStore {self.INDEX_DIR} klasöründen yüklendi")
                return vectorstore
            else:
                print(f"❌ {self.INDEX_DIR} klasörü bulunamadı veya boş")
                return None
        except Exception as e:
            print(f"❌ VectorStore yükleme hatası: {e}")
            return None
    
    def _create_vectorstore(self):
        """Yeni vectorstore oluşturur"""
        print("🔄 Yeni vectorstore oluşturuluyor...")
        
        # Load CSV data
        df = pd.read_csv(self.csv_file)
        documents = []
        
        print(f"📊 {len(df)} satır işleniyor...")
        
        for idx, row in df.iterrows():
            if idx % 10 == 0:
                print(f"İşlenen: {idx}/{len(df)}", end='\r')
            
            # Create text content
            text_content = self._create_text_content(row)
            
            # Download image and generate description
            image_url = row.get('link')
            image_description = ""
            
            if pd.notna(image_url):
                image = self._download_image(image_url)
                if image:
                    image_description = self._generate_image_description(image)
            
            # Create combined content
            combined_content = text_content
            if image_description:
                combined_content += f" | Image description: {image_description}"
            
            # Create metadata
            metadata = row.to_dict()
            metadata['doc_id'] = str(idx)
            metadata['image_description'] = image_description
            metadata['image_url'] = image_url
            
            # Create LangChain Document
            doc = Document(
                page_content=combined_content,
                metadata=metadata
            )
            documents.append(doc)
        
        print(f"\n✅ {len(documents)} LangChain dökümanı oluşturuldu!")
        
        # Create FAISS VectorStore
        print("🔄 FAISS VectorStore oluşturuluyor...")
        vectorstore = FAISS.from_documents(documents, self.embeddings_model)
        
        print(f"✅ VectorStore oluşturuldu! Döküman sayısı: {vectorstore.index.ntotal}")
        
        return vectorstore
    
    def _setup_vectorstore(self):
        """VectorStore'u yükler veya oluşturur"""
        print("✅ Yardımcı fonksiyonlar hazır!")
        
        self.vectorstore = self._load_vectorstore()
        
        if self.vectorstore is None:
            self.vectorstore = self._create_vectorstore()
            # Save the newly created vectorstore
            self._save_vectorstore()
        else:
            print(f"✅ Mevcut VectorStore yüklendi! Döküman sayısı: {self.vectorstore.index.ntotal}")
        
        print("✅ VectorStore hazır!")
    
    def _set_next_doc_id(self):
        """Yeni dökümanlar için bir sonraki ID'yi ayarlar"""
        if self.vectorstore is not None:
            # Mevcut en yüksek doc_id'yi bul
            try:
                max_id = 0
                # Vectorstore'daki tüm dökümanları kontrol et
                all_docs = self.vectorstore.similarity_search("", k=self.vectorstore.index.ntotal)
                for doc in all_docs:
                    doc_id = doc.metadata.get('doc_id', '0')
                    try:
                        id_num = int(doc_id)
                        max_id = max(max_id, id_num)
                    except ValueError:
                        continue
                self.next_doc_id = max_id + 1
            except:
                self.next_doc_id = 0
        else:
            self.next_doc_id = 0
    
    def add_data(self, 
                 text_data: Union[str, Dict[str, Any], List[Dict[str, Any]]], 
                 image_data: Union[str, Image.Image, List[Union[str, Image.Image]], None] = None,
                 metadata: Union[Dict[str, Any], List[Dict[str, Any]], None] = None) -> bool:
        
        if self.vectorstore is None:
            print("❌ VectorStore mevcut değil!")
            return False
        
        try:
            # Input'ları normalize et
            text_list = self._normalize_text_input(text_data)
            image_list = self._normalize_image_input(image_data, len(text_list))
            metadata_list = self._normalize_metadata_input(metadata, len(text_list))
            
            documents = []
            
            print(f"🔄 {len(text_list)} veri işleniyor...")
            
            for i, (text_item, image_item, meta_item) in enumerate(zip(text_list, image_list, metadata_list)):
                print(f"İşlenen: {i+1}/{len(text_list)}", end='\r')
                
                # Metin içeriğini hazırla
                if isinstance(text_item, dict):
                    text_content = self._dict_to_text(text_item)
                else:
                    text_content = str(text_item)
                
                # Görsel açıklamasını üret
                image_description = ""
                image_url = ""
                
                if image_item is not None:
                    if isinstance(image_item, Image.Image):  # PIL Image
                        image_description = self._generate_image_description(image_item)
                    elif isinstance(image_item, str):  # URL (sadece metadata için saklayalım)
                        image_url = image_item
                        # URL'den görsel indirmiyoruz, çünkü zaten PIL objesi verilecek
                
                # İçeriği birleştir
                combined_content = text_content
                if image_description:
                    combined_content += f" | Image description: {image_description}"
                
                # Metadata'yı hazırla
                final_metadata = {
                    'doc_id': str(self.next_doc_id + i),
                    'image_description': image_description,
                    'image_url': image_url,
                    'data_type': 'text_only' if not image_description else 'text_image',
                    'added_manually': True
                }
                
                # Kullanıcı metadata'sını ekle
                if meta_item:
                    final_metadata.update(meta_item)
                
                # Text data dict ise, onun alanlarını da metadata'ya ekle
                if isinstance(text_item, dict):
                    for key, value in text_item.items():
                        if key not in final_metadata:
                            final_metadata[key] = value
                
                # Document oluştur
                doc = Document(
                    page_content=combined_content,
                    metadata=final_metadata
                )
                documents.append(doc)
            
            # VectorStore'a ekle
            self.vectorstore.add_documents(documents)
            
            # ID sayacını güncelle
            self.next_doc_id += len(documents)
            
            # Kaydet
            self._save_vectorstore()
            
            print(f"\n✅ {len(documents)} veri başarıyla eklendi!")
            return True
            
        except Exception as e:
            print(f"❌ Veri ekleme hatası: {e}")
            return False
    
    def _normalize_text_input(self, text_data):
        """Text input'unu normalize eder"""
        if isinstance(text_data, str):
            return [text_data]
        elif isinstance(text_data, dict):
            return [text_data]
        elif isinstance(text_data, list):
            return text_data
        else:
            return [str(text_data)]
    
    def _normalize_image_input(self, image_data, target_length):
        """Image input'unu normalize eder - sadece PIL Image desteği"""
        if image_data is None:
            return [None] * target_length
        elif isinstance(image_data, Image.Image):
            # Tek PIL Image objesi - tüm text'ler için aynı resmi kullan
            return [image_data] * target_length
        elif isinstance(image_data, list):
            # PIL Image listesi - her text için ayrı görsel
            if len(image_data) == target_length:
                return image_data
            elif len(image_data) < target_length:
                # Eksik olanları None ile doldur
                return image_data + [None] * (target_length - len(image_data))
            else:
                # Fazla olanları kes
                return image_data[:target_length]
        else:
            return [None] * target_length
    
    def _normalize_metadata_input(self, metadata, target_length):
        """Metadata input'unu normalize eder"""
        if metadata is None:
            return [{}] * target_length
        elif isinstance(metadata, dict):
            return [metadata] * target_length
        elif isinstance(metadata, list):
            # Liste uzunluğunu target_length'e eşitle
            if len(metadata) == target_length:
                return metadata
            elif len(metadata) < target_length:
                # Eksik olanları boş dict ile doldur
                return metadata + [{}] * (target_length - len(metadata))
            else:
                # Fazla olanları kes
                return metadata[:target_length]
        else:
            return [{}] * target_length
    
    def _dict_to_text(self, data_dict):
        """Dictionary'yi text formatına çevirir"""
        text_parts = []
        for key, value in data_dict.items():
            if pd.notna(value):
                text_parts.append(f"{key}: {value}")
        return " | ".join(text_parts)
    
    def similarity_search(self, query: str, k: int = 3):
        """Benzer dökümanları arar"""
        if self.vectorstore is None:
            print("❌ VectorStore mevcut değil!")
            return []
        
        try:
            similar_docs = self.vectorstore.similarity_search(query, k=k)
            return similar_docs
        except Exception as e:
            print(f"❌ Arama hatası: {e}")
            return []
    
    def similarity_search_with_score(self, query: str, k: int = 3):
        """Benzer dökümanları skorlarıyla birlikte arar"""
        if self.vectorstore is None:
            print("❌ VectorStore mevcut değil!")
            return []
        
        try:
            similar_docs = self.vectorstore.similarity_search_with_score(query, k=k)
            return similar_docs
        except Exception as e:
            print(f"❌ Arama hatası: {e}")
            return []
    
    def get_vectorstore_info(self):
        """VectorStore hakkında bilgi döner"""
        if self.vectorstore is None:
            return "VectorStore mevcut değil"
        
        return {
            "document_count": self.vectorstore.index.ntotal,
            "next_doc_id": self.next_doc_id,
            "index_dir": self.INDEX_DIR,
            "embedding_model": self.EMBEDDING_MODEL,
            "image_model": self.IMAGE_MODEL
        }
    
    def add_documents(self, documents):
        """VectorStore'a yeni dökümanlar ekler (eski fonksiyon - backward compatibility)"""
        if self.vectorstore is None:
            print("❌ VectorStore mevcut değil!")
            return False
        try:
            self.vectorstore.add_documents(documents)
            self._save_vectorstore()
            print(f"✅ {len(documents)} döküman eklendi!")
            return True
        except Exception as e:
            print(f"❌ Döküman ekleme hatası: {e}")
            return False