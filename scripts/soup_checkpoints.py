#!/usr/bin/env python3
"""Average the weights of several checkpoints from the same training run
(weight soup). Both the raw `model` state and the `ema` state are averaged
when present; everything else (config/optimizer/...) is taken from the last
checkpoint so the result loads exactly like a regular checkpoint.

Usage:
    python scripts/soup_checkpoints.py out.pth in1.pth in2.pth [in3.pth ...]
"""

import sys

import torch


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(2)
    out_path = sys.argv[1]
    in_paths = sys.argv[2:]

    ckpts = [torch.load(p, map_location='cpu') for p in in_paths]
    models = [c.get('model', c) for c in ckpts]
    keys = models[0].keys()
    for m in models[1:]:
        assert m.keys() == keys, 'checkpoint state_dict keys differ'

    avg = {}
    for k in keys:
        vals = [m[k].float() for m in models]
        v = torch.stack(vals).mean(0)
        avg[k] = v.to(models[0][k].dtype)

    soup = dict(ckpts[-1])  # config/global_step/etc. from the last ckpt
    soup['model'] = avg
    soup['soup_sources'] = in_paths
    if all('ema' in c for c in ckpts):
        emas = [c['ema'].state_dict() if hasattr(c['ema'], 'state_dict') else c['ema'] for c in ckpts]
        try:
            ema_keys = emas[0].keys()
            soup['ema'] = {k: torch.stack([e[k].float() for e in emas]).mean(0).to(emas[0][k].dtype)
                           for k in ema_keys}
            print('EMA states averaged')
        except Exception as e:
            soup['ema'] = ckpts[-1].get('ema')
            print(f'EMA averaging skipped ({e}); using last checkpoint EMA')

    torch.save(soup, out_path)
    print(f'soup of {len(in_paths)} checkpoints -> {out_path}')


if __name__ == '__main__':
    main()
