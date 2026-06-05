import os
import sys
import tempfile
import logging
from concurrent import futures
import grpc

# Inject parent paths into sys.path to allow correct modules importing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.core.protos import vision_pb2, vision_pb2_grpc

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")


class VisionProcessor:
    """
    OCR processor using pytesseract (Tesseract OCR) on Windows.
    If Tesseract is not installed, logs a clear warning and returns empty string.
    """

    def __init__(self):
        self.tesseract_available = False
        try:
            import pytesseract
            import shutil

            # Search for Tesseract binary in common Windows locations
            tesseract_candidates = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                r"C:\msys64\ucrt64\bin\tesseract.exe",
                r"C:\msys64\mingw64\bin\tesseract.exe",
            ]

            tesseract_path = shutil.which("tesseract")
            if not tesseract_path:
                for candidate in tesseract_candidates:
                    if os.path.exists(candidate):
                        tesseract_path = candidate
                        break

            if tesseract_path:
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
                # Auto-detect and set TESSDATA_PREFIX for Tesseract to find language data
                tesseract_dir = os.path.dirname(tesseract_path)
                tessdata_candidates = [
                    os.path.join(tesseract_dir, "..", "share", "tessdata"),
                    os.path.join(tesseract_dir, "tessdata"),
                ]
                for tessdata in tessdata_candidates:
                    tessdata = os.path.normpath(tessdata)
                    if os.path.isdir(tessdata):
                        os.environ["TESSDATA_PREFIX"] = tessdata
                        break

            # Verify Tesseract is actually accessible
            pytesseract.get_tesseract_version()
            self.tesseract_available = True
            logger.info(f"VisionProcessor: Tesseract OCR initialized successfully ({tesseract_path}).")
        except Exception as e:
            logger.warning(
                f"VisionProcessor: Tesseract OCR is not available ({e}). "
                f"Screen capture OCR will be disabled. "
                f"Install Tesseract: https://github.com/tesseract-ocr/tesseract"
            )

    def process(self, image_path: str) -> str:
        """
        Perform real OCR on the given image file.
        Returns extracted text, or empty string if OCR is unavailable.
        """
        if not self.tesseract_available:
            return ""

        if not image_path or not os.path.exists(image_path):
            logger.warning(f"VisionProcessor: Image file not found: {image_path}")
            return ""

        try:
            import pytesseract
            from PIL import Image

            img = Image.open(image_path)
            # Try Vietnamese first, fallback to English
            try:
                text = pytesseract.image_to_string(img, lang="vie")
            except pytesseract.TesseractError:
                text = pytesseract.image_to_string(img, lang="eng")

            return text.strip()
        except Exception as e:
            logger.error(f"VisionProcessor: OCR failed for {image_path}: {e}")
            return ""


class VisionProcessorServicer(vision_pb2_grpc.VisionProcessorServicer):
    def __init__(self):
        """
        gRPC servicer that delegates OCR processing to pytesseract.
        """
        self.processor = VisionProcessor()

    def ExtractText(self, request, context):
        logger.info(f"gRPC Service: Received image extraction request (filename={request.filename})")

        # Save input bytes to temporary file for OCR reading
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(request.image_data)
            temp_path = tmp.name

        try:
            text = self.processor.process(temp_path)
            logger.info(f"gRPC Service: OCR completed. Extracted {len(text)} characters.")
            return vision_pb2.ExtractedTextResponse(
                text=text,
                metadata={"source": "grpc_worker", "chars_extracted": str(len(text))}
            )
        except Exception as e:
            logger.error(f"gRPC Service Error processing image: {e}")
            return vision_pb2.ExtractedTextResponse(
                text="",
                metadata={"status": "error", "message": str(e)}
            )
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as cleanup_err:
                    logger.warning(f"Failed to delete temp file {temp_path}: {cleanup_err}")


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
    vision_pb2_grpc.add_VisionProcessorServicer_to_server(VisionProcessorServicer(), server)
    server.add_insecure_port("[::]:50051")
    logger.info("Initializing gRPC Server on port 50051...")
    server.start()
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Stopping gRPC Server...")
        server.stop(0)


if __name__ == "__main__":
    serve()
