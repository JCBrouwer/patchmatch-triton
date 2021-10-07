"""
Pei Yu, Xiaokang Yang, and Li Chen, "Parallel-Friendly Patch Match Based on Jump Flooding"
https://www.researchgate.net/publication/278703228_Parallel-Friendly_Patch_Match_Based_on_Jump_Flooding

calculate Nearest Neighbor Field, f: A -> R^2.
for each patch coordinate in image A, difference between neareast patch in B, f(a) = b - a
D(v) is distance between patch (x,y) in A and (x,y)+v in B.
kNNF is multi-valued with k offset values for the k nearest patches

based on Jump Flooding, compute approximation to Voronoi diagram
for l in [n,n/2,n/4,...]:
    (x,y) passes information to (x+w,y+h) with w,h in {-l,0,l}

initialization
    offsets to random, independent, uniform sampled patches from B
    build max-heap for storing patch distances D of kNN in A
    build hash-table of kNNs to avoid re-testing patches already in kNN

propagation
    for l in [n,n/2,n/4,...]
        f_1(x,y) = argmin{D(f_1(x,y)), D(f_i(x+w,y+h))} for all i,w,h with w,h in {-l,0,l}
        where f_1(x,y) is worst-matched patch in max-heap (new version sifted down lower to max-heap)

parallelization
    each thread in kernel computes kNN of one patch in A
    each kernel launch completes on propagation of length l
    need to store 2 kNN heaps: former round, current round
    each thread queries offset in former heap, compares to current heap, and (maybe) sifts down in current
    copy current -> former after each round
"""
from time import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor


@torch.jit.script
def siftdown(heap: torch.Tensor, startpos: int, pos: int):
    newitem = heap[pos].clone()
    while pos > startpos:
        parentpos = (pos - 1) >> 1
        parent = heap[parentpos]
        if parent[0] < newitem[0]:
            heap[pos] = parent
            pos = parentpos
            continue
        break
    heap[pos] = newitem


@torch.jit.script
def siftup(heap: torch.Tensor, pos: int):
    endpos = heap.shape[0]
    startpos = pos
    newitem = heap[pos].clone()
    childpos = 2 * pos + 1
    while childpos < endpos:
        rightpos = childpos + 1
        if rightpos < endpos and not heap[rightpos][0] < heap[childpos][0]:
            childpos = rightpos
        heap[pos] = heap[childpos]
        pos = childpos
        childpos = 2 * pos + 1
    heap[pos] = newitem
    siftdown(heap, startpos, pos)


@torch.jit.script
def heapreplace(heap: torch.Tensor, item: torch.Tensor):
    heap[0] = item
    siftup(heap, 0)
    return heap


@torch.jit.script
def heapify(x: torch.Tensor):
    for i in torch.arange(x.shape[0] // 2, 0, -1):
        siftup(x, i)
    return x


@torch.jit.script
def distance(a: torch.Tensor, b: torch.Tensor):
    return torch.sum(torch.square(b - a), dim=[1, 2, 3])


# @torch.jit.script  # faster without script?!
def patch_match(img_A: torch.Tensor, img_B: torch.Tensor, K: int = 7, patch_size: int = 5):
    (A_h, A_w), (B_h, B_w) = img_A.shape[2:], img_B.shape[2:]
    r = patch_size // 2
    patch_range = torch.arange(-r, r + 1)
    patch_window = torch.stack(torch.meshgrid(patch_range, patch_range, indexing="ij"))[None]

    img_A = F.pad(img_A, (r, r, r, r), mode="reflect")
    img_B = F.pad(img_B, (r, r, r, r), mode="reflect")

    # initialize
    idxsB = torch.randint(B_h * B_w, size=(A_h, A_w, K))
    ysB = torch.div(idxsB, B_w, rounding_mode="floor") + r
    xsB = idxsB % B_w + r
    patch_ysB = (ysB[..., None, None] + patch_window[None, :, 0]).long()
    patch_xsB = (xsB[..., None, None] + patch_window[None, :, 1]).long()
    patchesB = img_B.squeeze()[:, patch_ysB, patch_xsB].permute(1, 2, 3, 4, 5, 0).reshape(A_h, A_w, K, -1)

    idxsA = torch.stack(torch.meshgrid(torch.arange(A_h), torch.arange(A_w), indexing="ij"))
    patchesA = idxsA[..., None, None] + patch_window[0, :, None, None]
    patchesA = img_A.squeeze()[:, patchesA[0], patchesA[1]].permute(1, 2, 3, 4, 0)
    patchesA = torch.tile(patchesA[:, :, None], [1, 1, K, 1, 1, 1]).reshape(A_h, A_w, K, -1)

    dAB = torch.sum(torch.square(patchesB - patchesA), dim=-1)
    kNNF = torch.stack((dAB, ysB, xsB), dim=-1)
    for y in torch.arange(A_h):
        for x in torch.arange(A_w):
            kNNF[y, x] = heapify(kNNF[y, x])

    # propagate
    max_side = torch.maximum(torch.tensor(A_h), torch.tensor(A_w))
    ls = torch.floor(max_side / torch.pow(2, torch.arange(torch.log2(max_side))))
    for l in ls:

        offsets = torch.stack([-l, torch.zeros([1]).squeeze(), l]).long()
        hs, ws = torch.meshgrid(offsets, offsets, indexing="ij")
        hs, ws = hs.flatten(), ws.flatten()

        for y in torch.arange(A_h):
            for x in torch.arange(A_w):

                ys, xs = (y + hs).clamp(0, A_h - 1).long(), (x + ws).clamp(0, A_w - 1).long()
                candidates = kNNF[ys, xs][:, :, [1, 2]].clone().reshape(hs.shape[0] * K, 2, 1, 1).long()
                idxs = candidates + patch_window

                # TODO how to use hashmap?

                dists = distance(
                    torch.tile(img_A[:, :, y : y + patch_size, x : x + patch_size], (idxs.shape[0], 1, 1, 1)),
                    img_B.squeeze()[:, idxs[:, 0], idxs[:, 1]].permute(1, 0, 2, 3),
                )
                best = torch.argmin(dists)
                best_candidate = candidates.squeeze()[best]

                kNNF[y, x] = heapreplace(kNNF[y, x], torch.stack([dists[best], best_candidate[0], best_candidate[1]]))

    return kNNF


if __name__ == "__main__":
    with torch.inference_mode():
        img_A = Image.open("bike_a.png")
        img_B = Image.open("bike_b.png").resize((img_A.size[1] + 25, img_A.size[0] - 12))

        img_A = to_tensor(img_A).unsqueeze(0)
        img_B = to_tensor(img_B).unsqueeze(0)

        P = 3
        K = 5

        t = time()
        kNNF = patch_match(img_A, img_B, K=K, patch_size=P)
        print(time() - t)

        result = torch.zeros((img_A.shape[2], img_A.shape[3], img_A.shape[1]))
        all_dists = torch.zeros((img_A.shape[2], img_A.shape[3]))
        for y in range(img_A.shape[2]):
            for x in range(img_A.shape[3]):
                all_dists[y, x] = kNNF[y, x, 0, 0]
                result[y, x] = img_B[:, :, kNNF[y, x, 0, 1].long() - P // 2, kNNF[y, x, 0, 2].long() - P // 2].squeeze()
        result = (result.cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(result).save(f"results/bike_{torch.mean(all_dists):.5f}.jpg")
