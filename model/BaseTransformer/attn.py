import torch
import torch.nn as nn
from einops import rearrange


class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        inner_dim = args.dim_head * args.heads
        project_out = not (args.heads == 1 and args.dim_head == args.dim)

        self.heads = args.heads
        self.scale = args.dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(args.dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, args.dim), nn.Dropout(args.dropout)) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out), attn


class AgentAwareAttention(nn.Module):
    def __init__(self, args, num_agent=200, num_control_features=9):
        super().__init__()
        in_len = args.in_len
        inner_dim = args.dim_head * args.heads
        project_out = not (args.heads == 1 and args.dim_head == args.dim)

        self.heads = args.heads
        self.scale = args.dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(args.dim, inner_dim * 3, bias=False)
        self.to_qk_self = nn.Linear(args.dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, args.dim), nn.Dropout(args.dropout)) if project_out else nn.Identity()

        attn_mask = torch.eye(num_agent)
        attn_mask = attn_mask.repeat_interleave(in_len, dim=1)
        attn_mask = attn_mask.repeat_interleave(in_len, dim=0)
        attn_mask = torch.cat([attn_mask, torch.zeros(attn_mask.size(0), num_control_features)], dim=1)
        attn_mask = torch.cat([attn_mask, torch.zeros(num_control_features, attn_mask.size(1))], dim=0)
        self.attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)

        """
        img_path = os.path.join("img", "inp_3types", "attention")
        os.makedirs(img_path, exist_ok=True)
        fig = plt.figure()
        plt.imshow(attn_mask, cmap="Blues")
        plt.colorbar()
        plt.savefig(f"img/inp_3types/attention/attention_mask_input_3types_lookback{self.ts_len}.png")
        """

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        qk_self = self.to_qk_self(x).chunk(2, dim=-1)
        q_self, k_self = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qk_self)

        attn_mask = self.attn_mask
        attn_mask = attn_mask.to(x.device)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        dots_self = torch.matmul(q_self, k_self.transpose(-1, -2)) * self.scale
        dots_all = attn_mask * dots_self + (1 - attn_mask) * dots

        attn = self.attend(dots_all)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        return self.to_out(out), attn
