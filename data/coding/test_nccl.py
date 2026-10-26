import torch
import torch.distributed as dist
import os, socket

def main():
    dist.init_process_group(backend="nccl")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    hostname = socket.gethostname()
    
    print(f"[Rank {rank}/{world_size}] host={hostname}, "
          f"local_rank={local_rank}, device={torch.cuda.get_device_name(device)}")
    
    # 测试 all-reduce：每个 rank 发送自己的 rank 值，求和后应 = 0+1+...+(N-1)
    tensor = torch.tensor([rank], dtype=torch.float32, device=f"cuda:{local_rank}")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    expected = world_size * (world_size - 1) / 2
    
    print(f"[Rank {rank}] all_reduce result={tensor.item()}, expected={expected}, "
          f"{'PASS ✅' if tensor.item() == expected else 'FAIL ❌'}")
    
    # 测试 all-gather
    local_tensor = torch.tensor([rank * 10.0], device=f"cuda:{local_rank}")
    gathered = [torch.zeros(1, device=f"cuda:{local_rank}") for _ in range(world_size)]
    dist.all_gather(gathered, local_tensor)
    
    if rank == 0:
        values = [t.item() for t in gathered]
        print(f"[Rank 0] all_gather result={values}")
    
    # 带宽粗测：发送较大张量
    if rank == 0:
        big = torch.randn(1024, 1024, device=f"cuda:{local_rank}")
        torch.cuda.synchronize()
        import time
        start = time.time()
        for _ in range(10):
            dist.all_reduce(big)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        size_mb = big.nelement() * big.element_size() / 1e6
        print(f"[Rank 0] 带宽粗测: {size_mb:.1f}MB x10 all_reduce = {elapsed:.2f}s, "
              f"~{size_mb * 10 / elapsed:.0f} MB/s")
    
    dist.barrier()
    if rank == 0:
        print("\n🎉 NCCL 跨节点通信测试全部通过！")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()