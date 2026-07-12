"""Patch train_activity_mt.py to add a --warm-start flag (transfer an encoder
from a pretrained .pt). Idempotent; safe to run once."""
import io
p = "src/train_activity_mt.py"
s = io.open(p, encoding="utf-8").read()

a_old = '    ap.add_argument("--threads", type=int, default=0)'
a_new = ('    ap.add_argument("--warm-start", default=None,\n'
         '                    help="pretrained .pt to transfer matching encoder weights from")\n'
         '    ap.add_argument("--threads", type=int, default=0)')

m_old = '    model = make_model(cfg, in_ch=in_ch).to(device)'
m_new = ('    model = make_model(cfg, in_ch=in_ch).to(device)\n'
         '    _ws = getattr(args, "warm_start", None)\n'
         '    if _ws:\n'
         '        import torch as _t\n'
         '        _sd = _t.load(_ws, map_location=device, weights_only=False)["state_dicts"][0]\n'
         '        _m = model.state_dict()\n'
         '        model.load_state_dict({k: v for k, v in _sd.items() if k in _m and _m[k].shape == v.shape}, strict=False)\n'
         '        print("    warm-started encoder from", _ws, flush=True)')

if "--warm-start" in s:
    print("already patched")
else:
    assert a_old in s, "argparse anchor not found"
    assert m_old in s, "model anchor not found"
    s = s.replace(a_old, a_new, 1).replace(m_old, m_new, 1)
    io.open(p, "w", encoding="utf-8").write(s)
    print("patched: added --warm-start to", p)
