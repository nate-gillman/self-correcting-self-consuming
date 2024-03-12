from typing import Dict, List

import torch

from sample.types import T_IDX_COLLATED_DIFFUSION_SAMPLE, DiffusionConditioningDict


def lengths_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(
        len(lengths), max_len
    ) < lengths.unsqueeze(1)
    return mask


def collate_tensors(batch: List[torch.Tensor]) -> torch.Tensor:
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate(batch: List[Dict[str, torch.Tensor]]) -> T_IDX_COLLATED_DIFFUSION_SAMPLE:
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b["inp"] for b in notnone_batches]
    if "lengths" in notnone_batches[0]:
        lenbatch = [b["lengths"] for b in notnone_batches]
    else:
        lenbatch = [len(b["inp"][0][0]) for b in notnone_batches]

    databatchTensor = collate_tensors(databatch)

    # the 'mask' property is derived from the length of sequences in the batch
    # and the final dimension of the motion sequences
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = (
        lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1])
        .unsqueeze(1)
        .unsqueeze(1)
    )  # unqueeze for broadcasting

    y: DiffusionConditioningDict = {"mask": maskbatchTensor, "lengths": lenbatchTensor}
    cond = {"y": y}

    if "text" in notnone_batches[0]:
        textbatch = [b["text"] for b in notnone_batches]
        cond["y"].update({"text": textbatch})

    if "tokens" in notnone_batches[0]:
        textbatch = [b["tokens"] for b in notnone_batches]
        cond["y"].update({"tokens": textbatch})

    if "action" in notnone_batches[0]:
        actionbatch = [b["action"] for b in notnone_batches]
        cond["y"].update({"action": torch.as_tensor(actionbatch).unsqueeze(1)})

    idx_tensor = None
    if "idxs" in notnone_batches[0]:
        # add the indexes of samples returned in this minibatch to the conditioning
        # information
        idxbatch = [b["idxs"] for b in notnone_batches]
        idx_tensor = torch.as_tensor(idxbatch).unsqueeze(1)

    # collate action textual names
    if "action_text" in notnone_batches[0]:
        action_text = [b["action_text"] for b in notnone_batches]
        cond["y"].update({"action_text": action_text})

    motion = databatchTensor
    return motion, cond, idx_tensor


# an adapter to our collate func
def t2m_collate(
    batch: torch.Tensor,
) -> T_IDX_COLLATED_DIFFUSION_SAMPLE:
    batch.sort(key=lambda x: x[0][3], reverse=True)

    adapted_batch = [
        {
            "inp": torch.tensor(b[4].T)
            .float()
            .unsqueeze(1),  # [seqlen, J] -> [J, 1, seqlen]
            "text": b[2],  # b[0]['caption']
            "tokens": b[6],
            "lengths": b[5],
            "idxs": idxs,
        }
        for b, idxs in batch
    ]
    return collate(adapted_batch)
