import os
import sys
import unittest
import time
import grpc
from unittest.mock import MagicMock, patch

# Ensure the app directories are importable
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from app.core.protos import vision_pb2, vision_pb2_grpc
from app.infrastructure.code_executor import PythonSandbox
from app.application.code_interpreter_usecase import CodeInterpreterUseCase
from app.infrastructure.vision_adapter import VisionAdapter
from app.application.rag_usecase import RAGUseCase
from app.infrastructure.vector_store import ChromaDBStore
from app.domain.interfaces import ILLMService


class TestDistributedRAGHub(unittest.TestCase):
    grpc_process = None

    @classmethod
    def setUpClass(cls):
        """
        Starts the gRPC vision server as a background subprocess if port 50051 is not bound,
        ensuring that test_grpc_vision_pipeline passes automatically in isolation.
        """
        import subprocess
        import socket
        import time
        
        # Test if port 50051 is already in use
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("127.0.0.1", 50051))
            s.close()
            # Port is free, start the gRPC vision server process
            python_exe = sys.executable
            server_script = os.path.join(os.path.dirname(__file__), "app", "workers", "vision_grpc_server.py")
            cls.grpc_process = subprocess.Popen([python_exe, server_script])
            # Give the server a moment to spin up and bind to the port
            time.sleep(2)
        except socket.error:
            # Port is already bound, assume server is running
            pass

    @classmethod
    def tearDownClass(cls):
        """
        Stops the gRPC vision server subprocess if we started it.
        """
        if cls.grpc_process:
            cls.grpc_process.terminate()
            cls.grpc_process.wait()

    def test_code_interpreter_sandbox(self):
        """
        Tests that PythonSandbox successfully executes python code and intercepts matplotlib charts.
        """
        sandbox = PythonSandbox()
        code = (
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "print('Hello Sandbox')\n"
            "plt.plot([1, 2], [3, 4])\n"
            "plt.show()\n"
        )
        res = sandbox.execute(code)
        if not res["success"]:
            print("[DEBUG] Sandbox execution failed! res =", res)
        self.assertTrue(res["success"])
        self.assertIn("Hello Sandbox", res["stdout"])
        self.assertGreater(len(res["charts"]), 0)
        self.assertTrue(res["charts"][0].startswith("/static/outputs/"))

    def test_grpc_vision_pipeline(self):
        """
        Tests the gRPC vision worker server and client communication.
        Verifies server is reachable and returns a valid response structure.
        """
        channel = grpc.insecure_channel("127.0.0.1:50051")
        try:
            # Check connection readiness
            grpc.channel_ready_future(channel).result(timeout=5)
            stub = vision_pb2_grpc.VisionProcessorStub(channel)
            request = vision_pb2.ImageRequest(
                image_data=b"\x89PNG\r\n\x1a\n",  # Minimal PNG header bytes
                filename="test_img.png"
            )
            response = stub.ExtractText(request, timeout=5)
            # Verify response structure is valid (text may be empty if Tesseract is not installed)
            self.assertIsNotNone(response.text)
            self.assertIsInstance(response.text, str)
            self.assertIn("source", response.metadata)
            self.assertIn("chars_extracted", response.metadata)
            print(f"[TEST INFO] gRPC OCR Response: {len(response.text)} chars extracted")
        except (grpc.FutureTimeoutError, grpc.RpcError) as e:
            self.fail(f"gRPC Vision Server on port 50051 is not responding. Error: {e}")

    def test_rag_no_web_fallback_when_disabled(self):
        """
        When search_web is disabled and local docs are irrelevant, the RAG pipeline
        must NOT call the web search (no silent hallucination from the internet).
        """
        mock_store = MagicMock()
        # Low-relevance result (distance > 1.3) so legacy code would have hit the web
        mock_store.search_similar.return_value = [
            {"text": "Irrelevant chunk", "metadata": {"source": "doc"}, "distance": 9.9}
        ]

        mock_llm = MagicMock()
        mock_llm.generate_answer.return_value = iter(["Tài liệu không đề cập."])

        rag = RAGUseCase(vector_store=mock_store, llm_service=mock_llm)

        with patch("app.infrastructure.web_search.FallbackWebSearch.search") as mock_web:
            events = list(rag.execute("Câu hỏi ngoài tài liệu", search_web=False))
            mock_web.assert_not_called()

        # Should still stream tokens and emit an empty citation list
        self.assertTrue(any(e.get("type") == "token" for e in events))

    def test_rag_web_search_when_enabled(self):
        """
        When search_web is enabled, the RAG pipeline must call the web search.
        """
        mock_store = MagicMock()
        mock_store.search_similar.return_value = []

        mock_llm = MagicMock()
        mock_llm.generate_answer.return_value = iter(["Web-based answer."])

        rag = RAGUseCase(vector_store=mock_store, llm_service=mock_llm)

        with patch("app.infrastructure.web_search.FallbackWebSearch.search") as mock_web:
            mock_web.return_value = "Some web context."
            list(rag.execute("Câu hỏi cần tra web", search_web=True))
            mock_web.assert_called_once()

    def test_rag_summary_instruction(self):
        """
        Tests that RAGUseCase system prompt contains strict summary instructions and does not force quiz generation on summary requests.
        """
        mock_store = MagicMock()
        mock_store.search_similar.return_value = [
            {"text": "Nội dung bài học có chứa từ quiz hoặc trắc nghiệm", "metadata": {"source": "doc.pdf"}, "distance": 0.2}
        ]

        mock_llm = MagicMock()
        mock_llm.generate_answer.return_value = iter(["Tóm tắt nội dung bài học kèm hình ảnh."])

        rag = RAGUseCase(vector_store=mock_store, llm_service=mock_llm)
        events = list(rag.execute("Tóm tắt nội dung kèm hình ảnh"))

        call_args = mock_llm.generate_answer.call_args
        system_prompt = call_args[1].get("system_prompt", "")
        self.assertIn("CRITICAL SUMMARY RULE", system_prompt)
        self.assertIn("ONLY generate a quiz if the user's explicit query", system_prompt)

    def test_document_deletion(self):
        """
        Tests that DeleteUseCase calls delete_document on vector store.
        """
        mock_store = MagicMock()
        from app.application.delete_usecase import DeleteUseCase
        use_case = DeleteUseCase(vector_store=mock_store)
        res = use_case.execute("test.pdf")
        
        self.assertTrue(res["success"])
        mock_store.delete_document.assert_called_once_with("test.pdf", "default")

    def test_fallback_llm_service(self):
        """
        Tests that FallbackLLMService falls back through the hierarchy:
        Mistral (fail) -> SambaNova (fail) -> Gemini (fail) -> Groq (success).
        """
        mock_mistral = MagicMock()
        mock_mistral.generate_answer.side_effect = Exception("Mistral Error")
        
        mock_sambanova = MagicMock()
        mock_sambanova.generate_answer.side_effect = Exception("SambaNova Error")
        
        mock_gemini = MagicMock()
        mock_gemini.generate_answer.side_effect = Exception("Gemini Error")
        
        mock_groq = MagicMock()
        mock_groq.generate_answer.return_value = ["Groq Response Chunk"]
        
        from app.infrastructure.fallback_llm import FallbackLLMService
        fallback_service = FallbackLLMService(
            groq_adapter=mock_groq, 
            gemini_adapter=mock_gemini,
            openrouter_adapter=None,
            sambanova_adapter=mock_sambanova,
            mistral_adapter=mock_mistral
        )
        
        tokens = list(fallback_service.generate_answer(context="Context", query="Query"))
        
        self.assertTrue(any("Groq Response Chunk" in t for t in tokens))
        mock_mistral.generate_answer.assert_called_once()
        mock_sambanova.generate_answer.assert_called_once()
        mock_gemini.generate_answer.assert_called_once()
        mock_groq.generate_answer.assert_called_once()

    def test_podcast_use_case(self):
        """
        Tests that PodcastUseCase generates dialogue script.
        """
        mock_store = MagicMock()
        mock_store.collection.count.return_value = 5
        mock_store.collection.get.return_value = {
            "documents": ["Chunk 1 content", "Chunk 2 content"],
            "metadatas": [{"source": "test.txt"}, {"source": "test.txt"}]
        }
        
        mock_llm = MagicMock()
        mock_llm.generate_answer.return_value = ['[{"host": "A", "text": "Hello world"}]']
        
        import asyncio
        from unittest.mock import AsyncMock

        from app.application.podcast_usecase import PodcastUseCase
        use_case = PodcastUseCase(vector_store=mock_store, llm_service=mock_llm)

        async def fake_tts(text, filepath, voice_key="A"):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "wb") as f:
                f.write(b"fake-mp3")
            return True

        with patch("app.application.podcast_usecase.synthesize_to_file", side_effect=fake_tts):
            res = asyncio.run(use_case.execute())
        
        self.assertTrue(res["success"])
        self.assertEqual(len(res["script"]), 1)
        self.assertEqual(res["script"][0]["host"], "A")
        self.assertEqual(res["script"][0]["text"], "Hello world")
        self.assertIn("audio_url", res)

    def test_synthesis_use_case(self):
        """
        Tests that SynthesisUseCase compiles notes.
        """
        mock_llm = MagicMock()
        mock_llm.generate_answer.return_value = ["Study Guide Result"]
        
        from app.application.synthesis_usecase import SynthesisUseCase
        use_case = SynthesisUseCase(llm_service=mock_llm)
        res = use_case.execute(notes=["Note 1 text", "Note 2 text"], action="study_guide")
        
        self.assertTrue(res["success"])
        self.assertEqual(res["result"], "Study Guide Result")
        mock_llm.generate_answer.assert_called_once()

    def test_red_teaming_use_case_chunking(self):
        """
        Tests RedTeamingUseCase with long content that triggers chunking.
        Verifies LLM prompts are called and deduplication handles findings correctly.
        """
        mock_store = MagicMock()
        mock_store.collection.get.return_value = {
            "documents": ["Long document chunk 1", "Long document chunk 2"],
            "metadatas": [{"chunk_index": 0}, {"chunk_index": 1}]
        }
        
        mock_llm = MagicMock()
        # Mocking two LLM calls (since there will be two chunks)
        # Chunk 1 returns normal findings, Chunk 2 returns duplicate finding and a mismatch category
        mock_llm.generate_answer.side_effect = [
            # Call 1 JSON response
            ['[\n  {\n    "text": "Trái Đất dẹt",\n    "category": "factual_error",\n    "explanation": "Trái Đất hình cầu dẹt hai cực.",\n    "confidence": 0.9\n  }\n]'],
            # Call 2 JSON response (one duplicate text, one mismatch category)
            ['[\n  {\n    "text": "Trái Đất dẹt",\n    "category": "factual_error",\n    "explanation": "Duplicate",\n    "confidence": 0.5\n  },\n  {\n    "text": "Nước sôi ở 0 độ",\n    "category": "context_mismatch",\n    "explanation": "Mâu thuẫn nhiệt độ sôi.",\n    "confidence": 0.95\n  }\n]']
        ]
        
        from app.application.red_teaming_usecase import RedTeamingUseCase
        use_case = RedTeamingUseCase(vector_store=mock_store, llm_service=mock_llm)
        
        # Test text long enough to trigger chunking (length > 8000)
        # 9000 chars will be split into 2 chunks of 8000 chars due to overlap
        long_content = "A" * 9000
        
        res = use_case.analyze(notebook_id="default", filename="test.txt", content=long_content)
        
        self.assertTrue(res["success"])
        findings = res["findings"]
        
        # Deduplication should keep the first occurrence
        self.assertEqual(len(findings), 2)
        self.assertEqual(findings[0]["text"], "Trái Đất dẹt")
        self.assertEqual(findings[0]["category"], "factual_error")
        self.assertEqual(findings[0]["confidence"], 0.9)
        
        self.assertEqual(findings[1]["text"], "Nước sôi ở 0 độ")
        self.assertEqual(findings[1]["category"], "context_mismatch")
        self.assertEqual(findings[1]["confidence"], 0.95)

    def test_sandbox_app_templates(self):
        """
        Tests get_suggested_templates under AppGeneratorUseCase.
        """
        mock_store = MagicMock()
        mock_llm = MagicMock()
        
        # Test case 1: no datasets
        mock_store.collection.get.return_value = {
            "documents": ["Normal doc text"],
            "metadatas": [{"source": "test.txt"}]
        }
        
        from app.application.app_generator_usecase import AppGeneratorUseCase
        use_case = AppGeneratorUseCase(vector_store=mock_store, llm_service=mock_llm)
        templates = use_case.get_suggested_templates("default")
        
        # Should only contain 'doc' type templates except 'data_analyzer' which is always featured
        for t in templates:
            if t["id"] == "data_analyzer":
                self.assertEqual(t["type"], "data")
            else:
                self.assertEqual(t["type"], "doc")
            
        # Test case 2: with datasets
        mock_store.collection.get.return_value = {
            "documents": ["[DATASET] test.csv\nFile path for pandas analysis: path/to/file.csv"],
            "metadatas": [{"source": "test.csv", "is_dataset": True}]
        }
        templates_with_data = use_case.get_suggested_templates("default")
        
        # Should contain both 'doc' and 'data' type templates
        has_data_template = any(t["type"] == "data" for t in templates_with_data)
        self.assertTrue(has_data_template)


if __name__ == "__main__":
    unittest.main()

