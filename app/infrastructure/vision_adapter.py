import os
import logging
import asyncio
import grpc
from app.core.protos import vision_pb2, vision_pb2_grpc
from app.core.events import event_queue

logger = logging.getLogger(__name__)


class VisionAdapter:
    def __init__(self, grpc_server: str = "localhost:50051"):
        """
        gRPC client and consumer adapter.
        Integrates with event-driven queue and acts as the event consumer.
        """
        self.grpc_server = grpc_server

    async def start_event_consumer(self, vector_store):
        """
        Asynchronous consumer loop that listens to event_queue.
        Forward images to gRPC server and saves the OCR result to ChromaDB.
        """
        logger.info("VisionAdapter (Consumer): Event listener loop active.")
        while True:
            try:
                event = await event_queue.get()
                if event.get("type") == "SCREEN_CAPTURED":
                    logger.info("VisionAdapter (Consumer): Dequeued SCREEN_CAPTURED event. Dispatching to gRPC worker...")
                    
                    # Establish async channel to gRPC service
                    async with grpc.aio.insecure_channel(self.grpc_server) as channel:
                        stub = vision_pb2_grpc.VisionProcessorStub(channel)
                        request = vision_pb2.ImageRequest(
                            image_data=event["image_data"],
                            filename=event["filename"]
                        )
                        
                        try:
                            response = await stub.ExtractText(request, timeout=10)
                            extracted_text = response.text
                            
                            if extracted_text and len(extracted_text.strip()) > 5:
                                metadata = {
                                    "source": "screen_capture",
                                    "timestamp": event["timestamp"]
                                }
                                # Store OCR text into vector database
                                vector_store.add_documents(
                                    texts=[extracted_text],
                                    metadatas=[metadata]
                                )
                                logger.info("VisionAdapter (Consumer): Indexed gRPC OCR text into vector store.")
                            else:
                                logger.debug("VisionAdapter (Consumer): OCR returned empty or short text, skipping indexing.")
                        except grpc.RpcError as rpc_err:
                            logger.error(f"gRPC remote service call failed: {rpc_err}")
                            
                event_queue.task_done()
            except Exception as e:
                logger.error(f"Error in VisionAdapter Consumer loop: {e}")
                await asyncio.sleep(2)

    def extract_text(self, image_path: str) -> str:
        """
        Synchronous method to extract text from an image via the gRPC vision server.
        Returns extracted text, or empty string on failure.
        """
        if not image_path or not os.path.exists(image_path):
            logger.warning(f"VisionAdapter: Image file not found: {image_path}")
            return ""

        try:
            with open(image_path, "rb") as f:
                image_data = f.read()

            with grpc.insecure_channel(self.grpc_server) as channel:
                stub = vision_pb2_grpc.VisionProcessorStub(channel)
                request = vision_pb2.ImageRequest(
                    image_data=image_data,
                    filename=os.path.basename(image_path)
                )
                response = stub.ExtractText(request, timeout=5)
                return response.text
        except Exception as e:
            logger.warning(f"VisionAdapter: gRPC OCR call failed: {e}")
            return ""
