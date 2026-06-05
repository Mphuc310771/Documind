import logging
from app.infrastructure.code_executor import PythonSandbox

logger = logging.getLogger(__name__)


class CodeInterpreterUseCase:
    def __init__(self, sandbox: PythonSandbox = None):
        """
        Application Service / Use Case designed under the CQRS pattern
        to execute code-interpreter data analysis commands.
        """
        self.sandbox = sandbox or PythonSandbox()

    def execute(self, code: str) -> dict:
        """
        Executes Python code in the sandbox and returns stdout, stderr, and chart image paths.
        """
        logger.info("Executing Python code interpreter command...")
        result = self.sandbox.execute(code)
        
        # Format a user-friendly console response
        status = "Success" if result["success"] else "Failed"
        logger.info(f"Code execution finished with status: {status}")
        
        return {
            "success": result["success"],
            "stdout": result["stdout"],
            "stderr": result["stderr"],
            "charts": result["charts"]
        }
