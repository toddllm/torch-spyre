import torch
import torch.nn as nn
from vllm_spyre.model_executor.model_loader.spyre import SpyreCausalLM
from fms.modules.attention import MultiHeadAttention


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = MultiHeadAttention(emb_dim=8, emb_kq=4, emb_v=4, nheads=2, kvheads=2)


class Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Block(), Block()])


class Dummy(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = Base()


obj = SpyreCausalLM.__new__(SpyreCausalLM)
nn.Module.__init__(obj)
obj.fms_model = Dummy()
obj.kv_cache_specs = {"num_layers": 2}
obj._attention_probe_state = None

SpyreCausalLM._stamp_attention_layers(obj)
attn0 = obj.fms_model.base_model.layers[0].attn
attn1 = obj.fms_model.base_model.layers[1].attn
assert attn0.layer_idx == 0
assert attn1.layer_idx == 1
assert attn0.layer_name.endswith("base_model.layers.0.attn")
assert attn1.layer_name.endswith("base_model.layers.1.attn")

state = SpyreCausalLM._prepare_attention_probe_state(obj, torch.device("cpu"))
state.ready[1] = 1
state.coverage[1] = 7
state.phase[1] = 3
state2 = SpyreCausalLM._prepare_attention_probe_state(obj, torch.device("cpu"))
assert state is state2
assert state2.ready.tolist() == [0, 0]
assert state2.coverage.tolist() == [0, 0]
assert state2.phase.tolist() == [-1, -1]

snap = SpyreCausalLM.get_attention_probe_snapshot(obj)
assert snap is not None
assert snap.ready.tolist() == [0, 0]
assert snap.coverage.tolist() == [0, 0]
assert snap.phase.tolist() == [-1, -1]

print("spyre helper probe ok")
