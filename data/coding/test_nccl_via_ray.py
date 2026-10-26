import ray
import torch
import os

@ray.remote(num_gpus=1)
class NCCLWorker:
    def __init__(self, rank, world_size, master_addr, master_port):
        # 关键：禁用 P2P，解决 3090 peer access 问题
        os.environ["NCCL_P2P_DISABLE"] = "1"
        os.environ["NCCL_SHM_DISABLE"] = "1"
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["NCCL_DEBUG"] = "WARN"

        import socket
        self.hostname = socket.gethostname()
        self.rank = rank

        torch.cuda.set_device(0)
        torch.distributed.init_process_group(
            backend="nccl", rank=rank, world_size=world_size
        )

    def test_allreduce(self):
        tensor = torch.tensor([float(self.rank)], device="cuda")
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        expected = sum(range(torch.distributed.get_world_size()))
        passed = abs(tensor.item() - expected) < 1e-5
        return (
            f"[Rank {self.rank:2d} @ {self.hostname}] "
            f"all_reduce={tensor.item():.0f}, expected={expected}, "
            f"{'PASS ✅' if passed else 'FAIL ❌'}"
        )

    def cleanup(self):
        torch.distributed.destroy_process_group()
        return "done"

ray.init(address="auto")
world_size = 18
master_addr = "10.154.24.41"
master_port = 29500

print(f"正在创建 {world_size} 个 NCCL worker（已禁用 P2P）...")
workers = [
    NCCLWorker.remote(i, world_size, master_addr, master_port)
    for i in range(world_size)
]

print("正在执行 all_reduce 测试...")
results = ray.get([w.test_allreduce.remote() for w in workers])
for r in results:
    print(r)

ray.get([w.cleanup.remote() for w in workers])
ray.shutdown()
print("\n🎉 NCCL 跨节点通信测试完成！")
