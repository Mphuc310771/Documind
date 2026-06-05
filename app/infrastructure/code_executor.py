import os
import sys
import uuid
import logging
import subprocess

logger = logging.getLogger(__name__)


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
        # Unique file ID for this execution run
        run_id = uuid.uuid4().hex
        script_filename = f"temp_run_{run_id}.py"
        
        # Prepend non-interactive backend configuration & auto-save hook for matplotlib
        interceptor_code = (
            "import os\n"
            "import matplotlib\n"
            "matplotlib.use('Agg')\n"
            "import matplotlib.pyplot as plt\n\n"
            f"os.makedirs('{self.output_dir}', exist_ok=True)\n"
            "def auto_save_show(*args, **kwargs):\n"
            f"    filename = f'chart_{run_id}.png'\n"
            f"    filepath = os.path.join('{self.output_dir}', filename)\n"
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
