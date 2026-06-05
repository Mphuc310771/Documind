import hashlib
import logging
import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)


class ChromaDBStore:
    def __init__(self, persist_directory: str = "./chroma_db", model_name: str = "keepitreal/vietnamese-sbert"):
        """
        Initialize the persistent ChromaDB client and local embedding model using SentenceTransformers.
        """
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=self.persist_directory)

        # Initialize SentenceTransformer embedding function
        logger.info(f"Loading embedding model: {model_name} ...")
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
        logger.info("Embedding model loaded successfully.")

        # Get or create the vector collection
        self.collection = self.client.get_or_create_collection(
            name="rag_collection",
            embedding_function=self.embedding_function
        )

    def add_documents(self, texts: list[str], metadatas: list[dict] = None):
        """
        Embed and store chunks of text.
        """
        if not texts:
            return

        # Generate unique ids based on MD5 hashes of the text content
        ids = [hashlib.md5(text.encode('utf-8')).hexdigest() for text in texts]
        
        self.collection.upsert(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"Upserted {len(texts)} chunks into ChromaDB.")

    def search_similar(self, query: str, top_k: int = 3, notebook_id: str = "default") -> list[dict]:
        """
        Embed the user query and return the top_k most similar text chunks along with metadata and distance.
        Returns a list of dictionaries where each dict has keys 'text', 'metadata', and 'distance'.
        """
        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where={"notebook_id": notebook_id}
        )

        output = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if ("metadatas" in results and results["metadatas"]) else [None] * len(docs)
            dists = results["distances"][0] if ("distances" in results and results["distances"]) else [999.0] * len(docs)
            
            for doc, meta, dist in zip(docs, metas, dists):
                output.append({
                    "text": doc,
                    "metadata": meta or {},
                    "distance": dist
                })
                
        return output

    def delete_document(self, filename: str, notebook_id: str = "default"):
        """
        Delete all chunks associated with a specific filename in a notebook.
        """
        if self.collection.count() > 0:
            self.collection.delete(where={"$and": [{"source": filename}, {"notebook_id": notebook_id}]})
            logger.info(f"Deleted chunks for '{filename}' in notebook '{notebook_id}' from ChromaDB.")
