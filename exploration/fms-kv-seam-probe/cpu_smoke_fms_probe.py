import torch
from fms.modules.attention import MultiHeadAttention
import fms.utils.spyre.paged  # noqa: F401


def build_attn(compiled: bool):
    attn = MultiHeadAttention(emb_dim=8, emb_kq=4, emb_v=4, nheads=2, kvheads=2)
    attn.layer_idx = 1
    attn.eval()
    for p in attn.parameters():
        p.requires_grad_(False)
    if compiled:
        attn = torch.compile(attn, backend="inductor")
    return attn


def run(compiled: bool):
    torch.manual_seed(0)
    attn = build_attn(compiled)
    q = torch.randn(1, 2, 8)
    position_ids = torch.arange(2).unsqueeze(0)
    key_cache = torch.zeros(1, 4, 2, 4)
    value_cache = torch.zeros(1, 4, 2, 4)
    slot_mapping = torch.tensor([[0, 1]], dtype=torch.int64)
    ready = torch.zeros(4, dtype=torch.int64)
    coverage = torch.zeros(4, dtype=torch.int64)
    phase = torch.full((4,), -1, dtype=torch.int64)

    with torch.inference_mode():
        out, cache = attn(
            q=q,
            position_ids=position_ids,
            past_key_value_state=(key_cache, value_cache),
            use_cache=True,
            attn_name="spyre_paged_attn",
            slot_mapping=slot_mapping,
            block_table=None,
            kv_probe_ready=ready,
            kv_probe_coverage=coverage,
            kv_probe_phase=phase,
        )

    assert out.shape == (1, 2, 8)
    assert cache[0].shape == key_cache.shape
    assert ready.tolist() == [0, 1, 0, 0], ready.tolist()
    assert coverage.tolist() == [0, 2, 0, 0], coverage.tolist()
    assert phase.tolist() == [-1, 0, -1, -1], phase.tolist()
    print(f"compiled={compiled} ok ready={ready.tolist()} coverage={coverage.tolist()} phase={phase.tolist()}")


if __name__ == "__main__":
    run(False)
    run(True)
