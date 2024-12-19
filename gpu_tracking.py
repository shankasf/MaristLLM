import time
from pynvml import *

def display_real_time_gpu_memory(interval=1):
    try:
        # Initialize NVIDIA Management Library
        nvmlInit()
        device_count = nvmlDeviceGetCount()

        while True:
            print("\033[2J\033[H", end="", flush=True)  # Clear the terminal and move cursor to the top
            print("Real-Time GPU Memory Usage:")
            print("-" * 40)

            for i in range(device_count):
                handle = nvmlDeviceGetHandleByIndex(i)
                info = nvmlDeviceGetMemoryInfo(handle)
                total = info.total // 1024 ** 2
                free = info.free // 1024 ** 2
                used = info.used // 1024 ** 2
                print(f"GPU {i}: Total={total} MB | Used={used} MB | Free={free} MB", flush=True)
            
            print("-" * 40)
            print("Press Ctrl+C to stop.", flush=True)
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nReal-time monitoring stopped.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        nvmlShutdown()

if __name__ == "__main__":
    display_real_time_gpu_memory(interval=1)
