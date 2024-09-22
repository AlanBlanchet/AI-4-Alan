import torch


def masked_collator(batch: list[dict]):
    names = batch[0].keys()
    transposed = zip(*[b.values() for b in batch])
    res = {}
    for name, samples in zip(names, transposed):
        ex = samples[0]

        if isinstance(ex, torch.Tensor):
            lengths = [len(s) if s.ndim != 0 else 0 for s in samples]

            max_len = max(lengths)
            if min(lengths) == max_len:
                res[name] = torch.stack(samples)
            else:
                elem = torch.zeros(
                    (len(samples), max_len, *samples[0].shape[1:]),
                    dtype=samples[0].dtype,
                )
                mask = torch.zeros((len(samples), max_len), dtype=torch.bool)
                for i, ln in enumerate(lengths):
                    elem[i, :ln] = samples[i]
                    mask[i, :ln] = True
                res[name] = elem
                res[f"{name}_mask"] = mask
        elif isinstance(ex, dict):
            res[name] = masked_collator(samples)
        else:
            res[name] = samples

    return res
