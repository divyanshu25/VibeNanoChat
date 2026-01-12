#!/usr/bin/env python3
"""
GPU Heat Script - Keep all GPUs at ~90%+ utilization

This script runs continuous matrix multiplications on all available GPUs
to maintain high GPU utilization for testing, benchmarking, or keeping GPUs warm.

Usage:
    python scripts/keep_gpus_hot.py              # Heat all GPUs immediately
    python scripts/keep_gpus_hot.py --delay 2    # Wait 2 hours, then heat all GPUs
    python scripts/keep_gpus_hot.py 0 --delay 1  # Wait 1 hour, then heat GPU 0
    python scripts/keep_gpus_hot.py 0,1,2        # Heat GPUs 0, 1, and 2

    Or with make:
    make gpu-hot              # Heat all GPUs
    make gpu-hot GPUS=0       # Heat only GPU 0
    make gpu-hot GPUS=0,2     # Heat GPUs 0 and 2

Stop with Ctrl+C
"""

import argparse
import signal
import sys
import time
from datetime import datetime

import torch
import torch.nn as nn


class GPUHeater:
    """Keeps GPUs hot with continuous computation"""

    def __init__(self, target_utilization=0.90, gpu_ids=None, delay_hours=None):
        self.target_utilization = target_utilization
        self.running = True
        self.gpu_workers = []
        self.gpu_ids = gpu_ids  # None means all GPUs
        self.delay_hours = delay_hours  # Hours to wait before starting

        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        print("\n\n🛑 Shutdown signal received. Stopping GPU heating...")
        self.running = False

    def _create_workload(self, device, size=8192):
        """
        Create a computationally intensive workload for a GPU

        Args:
            device: torch device (cuda:0, cuda:1, etc.)
            size: Matrix size for computation (larger = more GPU usage)

        Returns:
            Tuple of (model, data) for computation
        """
        # Create large matrices for multiplication
        # This keeps GPU busy with CUDA cores
        matrix_a = torch.randn(size, size, device=device, dtype=torch.float32)
        matrix_b = torch.randn(size, size, device=device, dtype=torch.float32)

        # Create a simple neural network for tensor cores usage
        model = nn.Sequential(
            nn.Linear(size, size),
            nn.ReLU(),
            nn.Linear(size, size),
            nn.ReLU(),
            nn.Linear(size, size),
        ).to(device)

        input_tensor = torch.randn(256, size, device=device)

        return model, input_tensor, matrix_a, matrix_b

    def _heat_gpu(self, gpu_id):
        """
        Continuously run computations on a specific GPU

        Args:
            gpu_id: GPU index to heat up
        """
        device = torch.device(f"cuda:{gpu_id}")

        print(f"🔥 Starting heating on GPU {gpu_id}")

        try:
            # Create workload
            model, input_tensor, matrix_a, matrix_b = self._create_workload(device)

            iteration = 0
            while self.running:
                # Matrix multiplication (keeps CUDA cores busy)
                torch.matmul(matrix_a, matrix_b)

                # Neural network forward pass (keeps tensor cores busy)
                output = model(input_tensor)

                # Backward pass for additional GPU usage
                loss = output.sum()
                loss.backward()

                # Clear gradients
                model.zero_grad()

                iteration += 1

                # Occasional logging (every 100 iterations)
                if iteration % 100 == 0:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"[{timestamp}] GPU {gpu_id}: Iteration {iteration} complete")

                # Small sleep to prevent 100% utilization (target ~90-95%)
                time.sleep(0.001)

        except Exception as e:
            print(f"❌ Error on GPU {gpu_id}: {e}")
        finally:
            print(f"✅ GPU {gpu_id} heating stopped")

    def start(self):
        """Start heating specified GPUs (or all if not specified)"""

        if not torch.cuda.is_available():
            print("❌ No CUDA GPUs available!")
            return

        num_gpus = torch.cuda.device_count()

        # Determine which GPUs to heat
        if self.gpu_ids is None:
            gpus_to_heat = list(range(num_gpus))
        else:
            gpus_to_heat = self.gpu_ids
            # Validate GPU IDs
            for gpu_id in gpus_to_heat:
                if gpu_id >= num_gpus or gpu_id < 0:
                    print(
                        f"❌ Invalid GPU ID: {gpu_id}. Available GPUs: 0-{num_gpus-1}"
                    )
                    return

        print("=" * 80)
        print("🔥 GPU HEATING SCRIPT")
        print("=" * 80)
        print(f"📊 Target utilization: {self.target_utilization * 100:.0f}%")
        print(f"🎯 Total GPUs detected: {num_gpus}")
        print(f"🔥 GPUs to heat: {gpus_to_heat}")
        print()

        # Show info for all detected GPUs
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            status = "🔥 HEATING" if i in gpus_to_heat else "❄️  IDLE"
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB) - {status}")

        print()

        # Wait for delay if specified
        if self.delay_hours:
            delay_seconds = self.delay_hours * 3600
            print(f"⏳ Waiting {self.delay_hours} hour(s) before starting...")
            print("💡 Press Ctrl+C to cancel")
            print("=" * 80)

            start_wait = time.time()
            try:
                while self.running and (time.time() - start_wait) < delay_seconds:
                    remaining = delay_seconds - (time.time() - start_wait)
                    hours_left = remaining / 3600
                    if int(remaining) % 60 == 0:  # Print every minute
                        print(f"⏳ {hours_left:.2f} hours remaining until start...")
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Cancelled during wait")
                return

            if not self.running:
                return
            print("\n⏱️  Wait complete! Starting GPU heating now...")

        print("🚀 Starting GPU heating...")
        print("💡 Press Ctrl+C to stop")
        print("=" * 80)
        print()

        # Create a process/thread for each specified GPU
        import threading

        threads = []
        for gpu_id in gpus_to_heat:
            thread = threading.Thread(target=self._heat_gpu, args=(gpu_id,))
            thread.daemon = True
            thread.start()
            threads.append(thread)

        # Monitor and keep running indefinitely
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Keyboard interrupt received")
            self.running = False

        # Wait for all threads to finish
        print("\n⏳ Waiting for GPU workers to stop...")
        for thread in threads:
            thread.join(timeout=5.0)

        print()
        print("=" * 80)
        print("✅ All GPU heating stopped successfully")
        print("=" * 80)


def show_gpu_status():
    """Show current GPU status before starting"""
    try:
        import subprocess

        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
        )

        print("\n📊 Current GPU Status:")
        print("-" * 80)
        lines = result.stdout.strip().split("\n")
        for line in lines:
            parts = line.split(", ")
            if len(parts) >= 5:
                idx, name, util, mem_used, mem_total = parts[:5]
                print(
                    f"  GPU {idx}: {util}% utilization, {mem_used}/{mem_total} MB memory"
                )
        print("-" * 80)
        print()
    except Exception as e:
        print(f"⚠️  Could not get GPU status: {e}")


def parse_gpu_ids(args):
    """
    Parse GPU IDs from command line arguments

    Supports formats:
    - No args: heat all GPUs
    - Single ID: "0" or "1"
    - Multiple IDs: "0 1 2" or "0,1,2"

    Returns:
        List of GPU IDs or None for all GPUs
    """
    if not args:
        return None  # Heat all GPUs

    gpu_ids = []
    for arg in args:
        # Handle comma-separated IDs
        if "," in arg:
            gpu_ids.extend([int(x.strip()) for x in arg.split(",")])
        else:
            gpu_ids.append(int(arg))

    return gpu_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Keep specified GPUs at ~90%+ utilization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                  # Heat all GPUs immediately
  %(prog)s --delay 2        # Wait 2 hours, then heat all GPUs
  %(prog)s 0 --delay 1      # Wait 1 hour, then heat GPU 0
  %(prog)s 0,1,2            # Heat GPUs 0, 1, and 2
        """,
    )
    parser.add_argument(
        "gpus",
        nargs="*",
        help="GPU IDs to heat (space or comma separated). Leave empty for all GPUs.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=None,
        help="Hours to wait before starting (default: start immediately)",
    )
    parser.add_argument(
        "--utilization",
        type=float,
        default=0.90,
        help="Target GPU utilization (0.0-1.0, default: 0.90)",
    )

    args = parser.parse_args()

    # Parse GPU IDs
    try:
        gpu_ids = parse_gpu_ids(args.gpus)
    except ValueError as e:
        print(f"❌ Invalid GPU ID format: {e}")
        sys.exit(1)

    # Show current status
    show_gpu_status()

    # Start heating
    heater = GPUHeater(
        target_utilization=args.utilization, gpu_ids=gpu_ids, delay_hours=args.delay
    )
    heater.start()

    # Show final status
    print("\n🔍 Final GPU status:")
    show_gpu_status()
