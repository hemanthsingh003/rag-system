#!/usr/bin/env python3
"""Ollama Manager - Start, run, and auto-unload qwen3:4b with resource monitoring"""

import argparse
import subprocess
import time
import sys
import threading
import os
import re


class ResourceMonitor:
    """Monitor system resource usage"""
    
    @staticmethod
    def get_ollama_processes() -> list:
        """Get all Ollama-related processes"""
        import psutil
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline', [])
                if cmdline and 'ollama' in ' '.join(cmdline).lower():
                    processes.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return processes
    
    @staticmethod
    def get_system_resources() -> dict:
        """Get current system CPU and memory usage"""
        import psutil
        cpu_percent = psutil.cpu_percent(interval=0.5)
        mem = psutil.virtual_memory()
        
        return {
            'cpu_percent': cpu_percent,
            'mem_total_gb': mem.total / (1024**3),
            'mem_available_gb': mem.available / (1024**3),
            'mem_used_gb': mem.used / (1024**3),
            'mem_percent': mem.percent
        }
    
    @staticmethod
    def get_ollama_resources() -> dict:
        """Get Ollama-specific resource usage"""
        import psutil
        processes = ResourceMonitor.get_ollama_processes()
        
        total_rss = 0
        total_cpu = 0
        process_names = []
        
        for proc in processes:
            try:
                mem_info = proc.memory_info()
                total_rss += mem_info.rss
                total_cpu += proc.cpu_percent(interval=0.1)
                process_names.append(proc.info['name'])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        return {
            'process_count': len(processes),
            'process_names': process_names,
            'ram_used_gb': total_rss / (1024**3),
            'cpu_percent': total_cpu
        }
    
    @staticmethod
    def print_resources(label: str = "Current"):
        """Print resource usage summary"""
        sys_res = ResourceMonitor.get_system_resources()
        ollama_res = ResourceMonitor.get_ollama_resources()
        
        print(f"\n{'='*60}")
        print(f"📊 {label} Resource Usage")
        print(f"{'='*60}")
        
        print(f"\n🖥️  System Resources:")
        print(f"   CPU:       {sys_res['cpu_percent']:.1f}%")
        print(f"   RAM:       {sys_res['mem_used_gb']:.2f}GB / {sys_res['mem_total_gb']:.2f}GB ({sys_res['mem_percent']:.1f}%)")
        
        print(f"\n🦙 Ollama Processes: {ollama_res['process_count']}")
        if ollama_res['process_names']:
            print(f"   Processes: {', '.join(set(ollama_res['process_names']))}")
        print(f"   RAM Used:  {ollama_res['ram_used_gb']:.2f}GB")
        print(f"   CPU:       {ollama_res['cpu_percent']:.1f}%")
        
        print(f"{'='*60}\n")


class OllamaManager:
    def __init__(self, model: str = "qwen3:4b", timeout: int = 300, monitor: bool = True):
        self.model = model
        self.timeout = timeout
        self.unload_timer = None
        self.monitor = monitor

    def is_service_running(self) -> bool:
        """Check if Ollama service is running"""
        result = subprocess.run(
            ["pgrep", "-f", "ollama"],
            capture_output=True
        )
        return result.returncode == 0

    def start_service(self):
        """Start Ollama in background"""
        if self.is_service_running():
            print("✓ Ollama service already running")
            return

        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        for _ in range(10):
            time.sleep(1)
            if self.is_service_running():
                print("✓ Ollama service started")
                return
        
        print("✗ Failed to start Ollama service")
        sys.exit(1)

    def is_model_loaded(self) -> bool:
        """Check if model is currently loaded in Ollama"""
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True
        )
        return self.model in result.stdout

    def generate(self, prompt: str, system_prompt: str | None = None) -> str | None:
        """Generate response with qwen3:4b"""
        if not self.is_service_running():
            self.start_service()
            time.sleep(2)

        full_prompt = prompt
        if system_prompt:
            full_prompt = f"System: {system_prompt}\n\nUser: {prompt}"

        if self.monitor:
            try:
                import psutil
                ResourceMonitor.print_resources("Before Generation")
            except ImportError:
                print("⚠️  psutil not available. Run: pip install psutil")
                self.monitor = False

        start_time = time.time()
        
        result = subprocess.run(
            ["ollama", "run", self.model],
            input=full_prompt,
            capture_output=True,
            text=True,
            timeout=180
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode != 0:
            print(f"✗ Error: {result.stderr}")
            return None

        if self.monitor:
            try:
                import psutil
                ResourceMonitor.print_resources("After Generation")
                print(f"⏱️  Generation time: {elapsed:.2f}s")
            except ImportError:
                pass

        return result.stdout.strip()

    def unload_model(self):
        """Unload model from RAM"""
        if self.is_model_loaded():
            subprocess.run(
                ["ollama", "stop", self.model],
                capture_output=True
            )
            print(f"✓ Model {self.model} unloaded from RAM")
        else:
            print(f"✓ Model {self.model} not loaded")

    def _cancel_unload_timer(self):
        """Cancel pending unload timer"""
        if self.unload_timer and self.unload_timer.is_alive():
            self.unload_timer.cancel()

    def run_with_auto_unload(self, prompt: str, system_prompt: str | None = None) -> str | None:
        """Run generation and auto-unload after timeout"""
        self._cancel_unload_timer()
        
        response = self.generate(prompt, system_prompt)
        
        self.unload_timer = threading.Timer(self.timeout, self.unload_model)
        self.unload_timer.daemon = True
        self.unload_timer.start()
        
        return response


def main():
    parser = argparse.ArgumentParser(description="Ollama Manager for qwen3:4b")
    parser.add_argument("prompt", nargs="?", help="Prompt for LLM")
    parser.add_argument("-s", "--system", help="System prompt", default=None)
    parser.add_argument("-t", "--timeout", type=int, default=300, 
                       help="Seconds to wait before unloading model (default: 300)")
    parser.add_argument("--no-unload", action="store_true", 
                       help="Keep model loaded after generation")
    parser.add_argument("--no-monitor", action="store_true", 
                       help="Disable resource monitoring")
    parser.add_argument("--monitor-only", action="store_true",
                       help="Only show current resource usage")
    
    args = parser.parse_args()

    try:
        import psutil
        monitor = not args.no_monitor
    except ImportError:
        print("⚠️  psutil not available. Run: pip install psutil")
        monitor = False

    manager = OllamaManager(timeout=args.timeout, monitor=monitor)

    if args.monitor_only:
        ResourceMonitor.print_resources("Current")
        return

    if args.prompt:
        print(f"Using model: {manager.model}")
        print(f"Timeout: {args.timeout}s")
        
        if not args.no_unload:
            response = manager.run_with_auto_unload(args.prompt, args.system)
        else:
            manager.start_service()
            response = manager.generate(args.prompt, args.system)
        
        if response:
            print("\n" + "="*50)
            print(response)
            print("="*50)
            
            if not args.no_unload:
                print(f"\n⏳ Model will unload in {args.timeout}s...")
    else:
        print(f"Ollama Manager ready with {manager.model}")
        print(f"Timeout: {args.timeout}s")
        print("Type 'quit' or 'exit' to exit, or 'monitor' to check resources.\n")
        
        manager.start_service()
        
        while True:
            try:
                prompt = input(">>> ")
                if prompt.lower() in ["quit", "exit"]:
                    break
                if prompt.lower() == "monitor":
                    ResourceMonitor.print_resources()
                    continue
                if not prompt.strip():
                    continue
                    
                response = manager.generate(prompt, args.system)
                if response:
                    print("\n" + response + "\n")
                    
            except KeyboardInterrupt:
                break
            except EOFError:
                break
        
        if not args.no_unload:
            manager.unload_model()
        print("\nDone.")


if __name__ == "__main__":
    main()
