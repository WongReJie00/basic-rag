import subprocess
import os
import time

class OVMSManager:
    def __init__(self, ovms_bin_dir, ovms_lib_dir, config_path, port=8002):
        self.ovms_bin = os.path.join(ovms_bin_dir, "ovms")
        self.ovms_lib_dir = ovms_lib_dir
        self.config_path = config_path
        self.port = port
        self.process = None

    def start(self, wait_model_name=None, wait_timeout=120):
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = self.ovms_lib_dir
        cmd = [self.ovms_bin, "--rest_port", str(self.port), "--config_path", self.config_path]
        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        print(f"Started OVMS on port {self.port}")
        # Wait for server to start and model to load
        if wait_model_name:
            import requests
            start_time = time.time()
            url = f"http://localhost:{self.port}/v1/config"
            while time.time() - start_time < wait_timeout:
                try:
                    resp = requests.get(url, timeout=5)
                    if resp.status_code == 200:
                        config = resp.json()
                        if wait_model_name in config:
                            print(f"Model '{wait_model_name}' is loaded and available.")
                            return
                except Exception:
                    pass
                time.sleep(2)
            print(f"Timeout waiting for model '{wait_model_name}' to load in OVMS.")
        else:
            time.sleep(5)  # Default wait if no model specified

    def stop(self):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
            print("Stopped OVMS.")
            self.process = None
        else:
            print("No OVMS process running.")

    def is_running(self):
        return self.process and self.process.poll() is None
