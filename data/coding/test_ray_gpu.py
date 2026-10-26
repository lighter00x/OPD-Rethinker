import ray

ray.init(address="auto")

@ray.remote(num_gpus=1)
def gpu_task():
    import torch, socket
    hostname = socket.gethostname()
    gpu_id = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(gpu_id)
    a = torch.randn(2000, 2000, device="cuda")
    b = torch.randn(2000, 2000, device="cuda")
    c = (a @ b).sum().item()
    return f"{hostname} | GPU {gpu_id}: {gpu_name} | matmul_sum={c:.2f}"

futures = [gpu_task.remote() for _ in range(18)]
results = ray.get(futures)
for i, r in enumerate(results):
    print(f"  Task {i:2d} -> {r}")

ray.shutdown()
print("\n✅ Ray 成功调度任务到全部 18 张 GPU")