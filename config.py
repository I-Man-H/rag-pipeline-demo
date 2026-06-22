from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # LLM
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    llm_model: str = Field(default="gpt-4o-mini", env="LLM_MODEL")
    llm_temperature: float = Field(default=0.0, env="LLM_TEMPERATURE")

    # Embeddings
    embedding_model: str = Field(
        default="text-embedding-3-small", env="EMBEDDING_MODEL"
    )

    # ChromaDB
    chroma_persist_dir: str = Field(
        default="./data/chroma_db", env="CHROMA_PERSIST_DIR"
    )
    chroma_collection_name: str = Field(
        default="rag_documents", env="CHROMA_COLLECTION_NAME"
    )

    # Chunking
    chunk_size: int = Field(default=512, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=64, env="CHUNK_OVERLAP")

    # Retrieval
    retrieval_top_k: int = Field(default=5, env="RETRIEVAL_TOP_K")

    # API
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
