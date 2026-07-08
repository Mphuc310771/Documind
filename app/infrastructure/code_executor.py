import os
import sys
import uuid
import logging
import subprocess
import ast

logger = logging.getLogger(__name__)


class SecurityVisitor(ast.NodeVisitor):
    def __init__(self):
        self.is_safe = True
        self.error_message = ""
        self.allowed_modules = {
            'pandas', 'numpy', 'matplotlib', 'seaborn', 'openpyxl', 
            'json', 'math', 'datetime', 'tabulate', 'plt'
        }
        self.blocked_names = {
            'eval', 'exec', '__import__', 'open', 'compile', 
            'globals', 'locals', 'getattr', 'setattr', 'delattr',
            'os', 'sys', 'subprocess', 'shutil', 'socket', 'urllib',
            'requests', 'builtins', 'pty', 'ctypes'
        }
        
    def visit_Import(self, node):
        for alias in node.names:
            root_module = alias.name.split('.')[0]
            if root_module not in self.allowed_modules:
                self.is_safe = False
                self.error_message = f"Thao tác import thư viện '{alias.name}' bị chặn vì lý do bảo mật."
                return
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node):
        if node.module:
            root_module = node.module.split('.')[0]
            if root_module not in self.allowed_modules:
                self.is_safe = False
                self.error_message = f"Thao tác import từ '{node.module}' bị chặn vì lý do bảo mật."
                return
        self.generic_visit(node)
        
    def visit_Name(self, node):
        if node.id in self.blocked_names:
            self.is_safe = False
            self.error_message = f"Sử dụng hàm/biến nguy hiểm '{node.id}' bị chặn vì lý do bảo mật."
            return
        self.generic_visit(node)
        
    def visit_Attribute(self, node):
        if node.attr.startswith('__') or node.attr in self.blocked_names:
            self.is_safe = False
            self.error_message = f"Truy cập thuộc tính nguy hiểm '{node.attr}' bị chặn vì lý do bảo mật."
            return
        self.generic_visit(node)
        
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            val = node.value.lower()
            if ".." in val or "/etc" in val or ".env" in val or "app_data.db" in val or "chroma_db" in val:
                self.is_safe = False
                self.error_message = f"Đường dẫn hoặc chuỗi nguy hiểm '{node.value}' bị chặn vì lý do bảo mật."
                return
        self.generic_visit(node)

    def visit_Str(self, node):
        val = node.s.lower()
        if ".." in val or "/etc" in val or ".env" in val or "app_data.db" in val or "chroma_db" in val:
            self.is_safe = False
            self.error_message = f"Đường dẫn hoặc chuỗi nguy hiểm '{node.s}' bị chặn vì lý do bảo mật."
            return
        self.generic_visit(node)


class PythonSandbox:
    def __init__(self, output_dir: str = "app/static/outputs"):
        """
        Secure sandboxed execution of python code using a subprocess.
        Automatically intercepts matplotlib plots and redirects them to the static outputs directory.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def execute(self, code: str) -> dict:
        """
        Executes code and returns stdout, stderr, and a list of generated chart paths.
        """
        # Validate code security
        try:
            tree = ast.parse(code)
            visitor = SecurityVisitor()
            visitor.visit(tree)
            if not visitor.is_safe:
                logger.warning(f"Security validation failed for generated code: {visitor.error_message}")
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": f"Lỗi Bảo Mật (Security Error): {visitor.error_message}",
                    "charts": []
                }
        except SyntaxError as syntax_err:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Lỗi Cú Pháp (Syntax Error): {syntax_err}",
                "charts": []
            }

        # Unique file ID for this execution run
        run_id = uuid.uuid4().hex
        script_filename = f"temp_run_{run_id}.py"
        
        output_dir_escaped = self.output_dir.replace('\\', '\\\\')
        # Prepend non-interactive backend configuration & auto-save hook for matplotlib
        interceptor_code = (
            "import os\n"
            "import matplotlib\n"
            "matplotlib.use('Agg')\n"
            "import matplotlib.pyplot as plt\n\n"
            f"os.makedirs('{output_dir_escaped}', exist_ok=True)\n"
            "def auto_save_show(*args, **kwargs):\n"
            f"    filename = f'chart_{run_id}.png'\n"
            f"    filepath = os.path.join('{output_dir_escaped}', filename)\n"
            "    plt.savefig(filepath, bbox_inches='tight')\n"
            "    print(f'__CHART_SAVED__:{filename}')\n"
            "    plt.close()\n"
            "plt.show = auto_save_show\n\n"
        )
        
        full_code = interceptor_code + code
        
        # Write temporary script file
        with open(script_filename, "w", encoding="utf-8") as f:
            f.write(full_code)
            
        try:
            # Execute python inside the active virtual environment python executable if available
            python_exe = sys.executable
            result = subprocess.run(
                [python_exe, script_filename],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            stdout = result.stdout
            stderr = result.stderr
            
            # Parse saved charts from stdout tags
            charts = []
            cleaned_stdout_lines = []
            for line in stdout.splitlines():
                if line.startswith("__CHART_SAVED__:"):
                    chart_name = line.split("__CHART_SAVED__:")[1].strip()
                    # Return relative web path to static outputs
                    charts.append(f"/static/outputs/{chart_name}")
                else:
                    cleaned_stdout_lines.append(line)
                    
            cleaned_stdout = "\n".join(cleaned_stdout_lines)
            
            return {
                "success": result.returncode == 0,
                "stdout": cleaned_stdout,
                "stderr": stderr,
                "charts": charts
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": "Execution timed out (limit: 30s)",
                "charts": []
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "charts": []
            }
        finally:
            # Clean up temporary script file
            if os.path.exists(script_filename):
                try:
                    os.remove(script_filename)
                except Exception as e:
                    logger.warning(f"Failed to remove temp script file {script_filename}: {e}")
