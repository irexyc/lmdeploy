import sys
sys.path.insert(0, "/nvme1/chenxin/ws/gloo/build/lib")

import threading
import torch
from _moe import HostGroupId, MoeTest


def test_allreduce_sum(h_comm, rank, n_ranks):
    torch.cuda.set_device(rank)
    handle = MoeTest(h_comm, rank, n_ranks)
    ten = torch.ones(4, 4, dtype=torch.half, device='cuda') * rank
    handle.allreduce_sum(ten)
    print(f'{rank=}, {ten=}')

def test_reduce_scatter(h_comm, rank, n_ranks):
    torch.cuda.set_device(rank)
    handle = MoeTest(h_comm, rank, n_ranks)
    token = 4
    hidden = 4096
    ten = torch.ones(token * n_ranks, hidden, dtype=torch.half, device='cuda')
    for i in range(n_ranks):
        ten[i*token:(i+1)*token, :] *= i
    out = handle.reduce_scatter(ten, True)
    assert torch.sum(out == rank * n_ranks) == token * hidden, out


def test_all2all(h_comm, rank, n_ranks, barrier, shared_data):
    # settings
    token = 1024
    hidden = 7168
    num_experts = 256
    num_topk = 8
    # process
    torch.cuda.set_device(rank)
    torch.cuda.manual_seed(rank)
    handle = MoeTest(h_comm, rank, n_ranks)
    logits = torch.randn(token, hidden, dtype=torch.bfloat16, device='cuda')
    scores = torch.randn(token, num_experts, dtype=torch.float, device='cuda')

    scales = torch.zeros(num_experts, token, dtype=torch.float, device='cuda')
    masks = torch.full((num_experts, token), -1, dtype=torch.int8, device='cuda')
    token_idx_in_rank = torch.full((n_ranks, token), -1, dtype=torch.int32, device='cuda')
    topk_scores, topk_idx = torch.topk(scores, num_topk, dim=-1)
    topk_scores = topk_scores.softmax(dim=-1)
    scales.scatter_(0, topk_idx.t(), topk_scores.t())
    masks.scatter_(0, topk_idx.t(), 1)
    rank_idx = topk_idx // (num_experts // n_ranks)
    for i in range(n_ranks):
        mask = (rank_idx == i).sum(dim=-1) > 0
        token_idx_in_rank[i, mask] = torch.arange(mask.sum(), dtype=torch.int32, device='cuda')

    send_logits = [None] * n_ranks
    for i in range(n_ranks):
        send_logits[i] = logits[token_idx_in_rank[i] != -1]
    shared_data[rank] = send_logits
    barrier.wait()
    recv_logits = torch.cat([_data[rank].to(device=rank) for _data in shared_data], dim=0)
    ref_token_idx_in_rank, ref_recv_hidden, ref_logits = handle.all2all(logits, scores, num_topk)

    # check received logits
    assert torch.sum((recv_logits - ref_recv_hidden).abs()) == 0
    # print(f'{rank=}, {ref_logits / logits=}')




def main():
    h_comm = HostGroupId()
    h_comm.init()

    n_ranks = 8
    barrier = threading.Barrier(n_ranks)
    shared_data = [None] * n_ranks
    threads = []
    for rank in range(n_ranks):
        # t = threading.Thread(target=test_allreduce_sum, args=(h_comm, rank, n_ranks))
        # t = threading.Thread(target=test_reduce_scatter, args=(h_comm, rank, n_ranks))
        t = threading.Thread(target=test_all2all, args=(h_comm, rank, n_ranks, barrier, shared_data))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()


if __name__ == '__main__':
    main()
