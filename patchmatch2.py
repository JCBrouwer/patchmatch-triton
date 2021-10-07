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

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm


def heapreplace(heap, item):
    heap[0] = item
    siftup(heap, 0)


def heapify(x):
    for i in reversed(range(len(x) // 2)):
        siftup(x, i)


def siftdown(heap, startpos, pos):
    newitem = heap[pos]
    while pos > startpos:
        parentpos = (pos - 1) >> 1
        parent = heap[parentpos]
        if parent[0] < newitem[0]:
            heap[pos] = parent
            pos = parentpos
            continue
        break
    heap[pos] = newitem


def siftup(heap, pos):
    endpos = len(heap)
    startpos = pos
    newitem = heap[pos]
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


def patch_match_reconstruct(img_A, img_B, K=7, patch_size=3):

    img_A = to_tensor(img_A).unsqueeze(0)
    img_B = to_tensor(img_B).unsqueeze(0)

    A_c, A_h, A_w = img_A.shape[1:]
    B_h, B_w = img_B.shape[2:]
    r = patch_size // 2
    patch_range = torch.arange(-r, r + 1)
    patch_window = torch.stack(torch.meshgrid(patch_range, patch_range, indexing="ij"))[None]

    img_A = F.pad(img_A, (r, r, r, r), mode="reflect")
    img_B = F.pad(img_B, (r, r, r, r), mode="reflect")

    def distance(a, b):
        return torch.sum(torch.square(b - a), axis=[1, 2, 3])

    print("initializing")
    idxsA = torch.stack(torch.meshgrid(torch.arange(A_h), torch.arange(A_w), indexing="ij"))
    ysA, xsA = idxsA[..., None, None] + patch_window[0, :, None, None]
    patchesA = img_A.squeeze()[:, ysA, xsA].permute(1, 2, 3, 4, 0)
    patchesA = torch.tile(patchesA[:, :, None], [1, 1, K, 1, 1, 1]).reshape(A_h, A_w, K, -1)

    idxsB = torch.randint(B_h * B_w, size=(A_h, A_w, K))
    ysB = torch.div(idxsB, B_w, rounding_mode="floor") + r
    xsB = idxsB % B_w + r
    patch_ysB = ysB[..., None, None] + patch_window[None, :, 0]
    patch_xsB = xsB[..., None, None] + patch_window[None, :, 1]

    patchesB = img_B.squeeze()[:, patch_ysB, patch_xsB].permute(1, 2, 3, 4, 5, 0).reshape(A_h, A_w, K, -1)
    dAB = torch.sum(torch.square(patchesB - patchesA), axis=-1)
    kNNF = torch.stack((dAB, ysB, xsB), axis=-1)
    for y in torch.arange(A_h):
        for x in torch.arange(A_w):
            heapify(kNNF[y, x])

    print("propagating")
    for iter in range(5):

        l = max(A_h, A_w)
        while l >= 1:
            offsets = torch.tensor([-l, 0, l])
            hs, ws = torch.meshgrid(offsets, offsets, indexing="ij")
            hs, ws = hs.flatten(), ws.flatten()

            temp = np.zeros((A_h, A_w, A_c))
            all_dists = []

            for y in tqdm(torch.arange(A_h)):
                for x in torch.arange(A_w):
                    ys, xs = (y + hs).clamp(0, A_h - 1), (x + ws).clamp(0, A_w - 1)
                    candidates = kNNF[ys, xs][:, :, [1, 2]].reshape(len(hs) * K, 2, 1, 1).long()

                    idxs = candidates + patch_window

                    # TODO how to use hashmap?

                    dists = distance(
                        torch.tile(img_A[:, :, y : y + patch_size, x : x + patch_size], (len(idxs), 1, 1, 1)),
                        img_B.squeeze()[:, idxs[:, 0], idxs[:, 1]].permute(1, 0, 2, 3),
                    )
                    best = torch.argmin(dists)

                    print(
                        kNNF[y, x].squeeze().numpy(), torch.tensor((dists[best], *candidates.squeeze()[best])).numpy()
                    )
                    heapreplace(kNNF[y, x], torch.tensor((dists[best], *candidates.squeeze()[best])))

                    best_patch = kNNF[y, x, 0]
                    all_dists.append(best_patch[0])
                    yB, xB = best_patch[[1, 2]].long()
                    temp[y, x] = img_B[:, :, yB + r, xB + r].squeeze().cpu().numpy()

            Image.fromarray((temp * 255).astype(np.uint8)).save(f"{iter}_{l}_{np.mean(all_dists):.5f}.jpg")

            l //= 2


file_A = "PatchMatch-Pytorch/bike_a.png"
file_B = "PatchMatch-Pytorch/bike_b.png"
img_A = Image.open(file_A)
img_B = Image.open(file_B).resize((img_A.size[1] + 25, img_A.size[0] - 12))
patch_match_reconstruct(img_A, img_B)
