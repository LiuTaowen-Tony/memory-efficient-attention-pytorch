from dataclasses import dataclass
from transformers import HfArgumentParser
import torch
import torch.nn.functional as F
from torch import optim
from memory_efficient_attention_pytorch.memory_efficient_attention import Attention

# memory usage

@dataclass
class Args:
    heads: int
    seq_len: int
    dim_head: int
    casual: bool
    memory_efficient: bool
    q_bucket_size: int
    k_bucket_size: int


def main():
    parser = HfArgumentParser((Args))
    args = parser.parse_args_into_dataclasses()[0]

    torch.cuda.memory._record_memory_history()
    layer = Attention(**args.__dict__)
    opt = opt.Adam(layer.parameters(), lr=0.01)

    # forward pass
    for i in range(3):
        x = torch.randn(args.seq_len, args.dim_head * args.heads)
        y = torch.randn(args.seq_len, args.dim_head * args.heads)

        loss = F.mse_loss(layer(x), y)

        loss.backward()

        opt.step()

    torch.cuda.memory._dump_snapshot()
    

