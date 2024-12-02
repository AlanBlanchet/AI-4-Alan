from ...modality.modality import Modality


def masked_collator(batch: list[dict]):
    names = batch[0].keys()
    transposed = zip(*[b.values() for b in batch])

    items = {name: samples for name, samples in zip(names, transposed)}

    # Redirect to the modality collators
    return Modality.collate(items)
