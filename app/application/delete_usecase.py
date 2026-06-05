import logging
from app.infrastructure.vector_store import ChromaDBStore

logger = logging.getLogger(__name__)


class DeleteUseCase:
    def __init__(self, vector_store: ChromaDBStore):
        """
        Use case for deleting documents.
        """
        self.vector_store = vector_store

    def execute(self, filename: str, notebook_id: str = "default") -> dict:
        try:
            self.vector_store.delete_document(filename, notebook_id)
            return {
                "success": True,
                "message": f"Successfully deleted all chunks of '{filename}'."
            }
        except Exception as e:
            logger.error(f"Error deleting document '{filename}': {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Failed to delete document: {str(e)}"
            }
