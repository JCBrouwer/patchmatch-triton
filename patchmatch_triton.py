"""
Pei Yu, Xiaokang Yang, and Li Chen, "Parallel-Friendly Patch Match Based on Jump Flooding"
https://www.researchgate.net/publication/278703228_Parallel-Friendly_Patch_Match_Based_on_Jump_Flooding

calculate Nearest Neighbor Field, f: A -> R^2.
for each patch coordinate in image A, difference between neareast patch in B, f(a) = b - a
D(v) is distance between patch (x,y) in A and (x,y)+v in B.
kNNF is multi-valued with k offset values for the k nearest patches

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
    each kernel launch completes one propagation of length l
    need to store 2 kNN heaps: former round, current round (is this really true?)
    each thread queries offset in former heap, compares to current heap, and (maybe) sifts down in current
    copy current -> former after each round
"""

from time import time

import numpy as np
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from PIL import Image
from torchvision.transforms.functional import to_tensor

GPU = torch.device("cuda")


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


# fmt: off
@triton.jit
def propagate_kernel(A_ptr, B_ptr, kNNF_ptr, output_ptr, l: int, K: int, P: int, A_h: int, A_w: int, A_c: int, **meta):
    """
    Args:
        A_ptr : pointer to image A which has shape (A_w, A_h, A_c)
        B_ptr : pointer to image B
        kNNF_ptr : pointer to k nearest neighbor field which is (A_w , A_h, K, 3)
        output_ptr : pointer to output kNNF values for this iteration which is (A_w, A_h, K)
        l (int): looking distance (where to look for candidates)
        K (int): number of nearest neighbors (default: 5)
        P (int): patch size (default: 3)
        A_h (int): height of image A
        A_w (int): width of image A
        A_c (int): number of channels in image A
    """
    
    B = meta["BLOCK_SIZE"] # 16

    # map pid to block of kNNF that it should compute
    pid = tl.program_id(axis=0)
    num_pid_h = tl.cdiv(A_h, B)
    num_pid_w = tl.cdiv(A_w, B)
    y = pid // num_pid_h
    x = pid // num_pid_w
    ys = y * B + tl.arange(0, B)
    xs = x * B + tl.arange(0, B)
    A_grid = ys[:, None] + xs[None, :]  # TODO need to add stride in ptr sum?

    # arrays for indexing certain things
    window = tl.arange(0, 3) - P // 2  # tl.arange(-P // 2, P // 2 + 1)
    channels = tl.arange(0, 3)  # A_c 
    coord_dim = tl.arange(1, 3) 
    dist_dim = tl.zeros((1,), dtype=tl.int32)
    neighbors = tl.arange(0, 5)  # K

    # extract pixel value patches from A
    A_patch_center_idxs = A_ptr + A_grid  # B, B
    A_patch_idxs = (
        A_patch_center_idxs[   :,    :, None, None, None, None]
        + window           [None, None, None,    :, None, None]
        + window           [None, None, None, None,    :, None]
        + channels         [None, None, None, None, None,    :]
    )  #                    B     B     1     P     P     A_c
    A_patches = tl.load(A_patch_idxs)

    # look for new candidate patches l pixels left, right, up, and down (9 locations)
    candidates = tl.zeros((9, B, B, 2), dtype=tl.int32)
    candidate_distances = tl.zeros((9, B, B), dtype=tl.float32)
    i = 0
    for h in range(-l, l, l):
        for w in range(-l, l, l):

            # ensure indexes stay in bounds
            candidate_ys = tl.maximum(0, tl.minimum(A_h - 1, ys + h))
            candidate_xs = tl.maximum(0, tl.minimum(A_w - 1, xs + w))

            # load candidate patch centers from kNNF
            candidate_coords = (
                kNNF_ptr
                + candidate_ys[   :, None, None, None]
                + candidate_xs[None,    :, None, None]
                + neighbors   [None, None,    :, None]
                + coord_dim   [None, None, None,    :]
            )  #               B     B     K     2

            # candidate_coords shape here is actually (B, B, K, 3) !?
            # candidate_idxs = tl.sum(candidate_coords, axis=3)  # Error: Encountered unimplemented code path in sum. This is likely a bug on our side.
            candidate_idxs = candidate_coords[:, :, :, 0] + candidate_coords[:, :, :, 1]  # Error: cannot reshape block of different shape
            B_patch_center_idxs = tl.load(candidate_idxs)

            # load corresponding patches from image B
            B_patch_idxs = (
                B_ptr
                + B_patch_center_idxs[   :,    :,    :, None, None, None]
                + window             [None, None, None,    :, None, None]
                + window             [None, None, None, None,    :, None]
                + channels           [None, None, None, None, None,    :]
            )  #                      B     B     K     P     P     A_c
            B_patches = tl.load(B_patch_idxs)

            # find distance between image A patches and candidate patches
            distances = tl.sum((B_patches - A_patches) ** 2, axis=[3, 4, 5])  # B, B, K

            # remember best candidates
            best_candidates = triton.torch.argmin(distances, axis=2)  # B, B
            candidates[i] = B_patch_center_idxs[best_candidates]
            candidate_distances[i] = distances[best_candidates]

            i += 1

    # find overal best candidates
    idxs_new = triton.torch.argmin(candidate_distances, axis=0)  # B, B
    kNNF_coord_new = candidates[idxs_new[None, :, :, None]]  # B, B, 2
    kNNF_dist_new = candidate_distances[idxs_new[None, :, :]]  # B, B

    # store 
    tl.store(output_ptr + A_grid[:, :, None] + dist_dim[None, None, :], kNNF_dist_new)
    tl.store(output_ptr + A_grid[:, :, None] + coord_dim[None, None, :], kNNF_coord_new)
# fmt: on


def propagate(A: torch.Tensor, B: torch.Tensor, kNNF: torch.Tensor, l: int, K: int, P: int):
    _, A_c, A_h_pad, A_w_pad = A.shape
    A_h, A_w = A_h_pad - P, A_w_pad - P

    output = torch.empty((A_h, A_w, 3), device=kNNF.device, dtype=kNNF.dtype)

    grid = lambda meta: (triton.cdiv(A_h, meta["BLOCK_SIZE"]) * triton.cdiv(A_w, meta["BLOCK_SIZE"]),)

    pgm = propagate_kernel[grid](
        A.squeeze().permute(1, 2, 0),
        B.squeeze().permute(1, 2, 0),
        kNNF,
        output,
        int(l),
        int(K),
        int(P),
        int(A_h),
        int(A_w),
        int(A_c),
        BLOCK_SIZE=16,
    )

    return output


# @torch.jit.script  # faster without script?!
def patch_match(img_A: torch.Tensor, img_B: torch.Tensor, K: int = 7, P: int = 5):
    (A_h, A_w), (B_h, B_w) = img_A.shape[2:], img_B.shape[2:]
    r = P // 2
    patch_range = torch.arange(-r, r + 1).to(GPU)
    patch_window = torch.stack(torch.meshgrid(patch_range, patch_range, indexing="ij"))[None]

    img_A = F.pad(img_A, (r, r, r, r), mode="reflect")
    img_B = F.pad(img_B, (r, r, r, r), mode="reflect")

    # initialize
    idxsB = torch.randint(B_h * B_w, size=(A_h, A_w, K)).to(GPU)
    ysB = torch.div(idxsB, B_w, rounding_mode="floor") + r
    xsB = idxsB % B_w + r
    patch_ysB = (ysB[..., None, None] + patch_window[None, :, 0]).long()
    patch_xsB = (xsB[..., None, None] + patch_window[None, :, 1]).long()
    patchesB = img_B.squeeze()[:, patch_ysB, patch_xsB].permute(1, 2, 3, 4, 5, 0).reshape(A_h, A_w, K, -1)

    idxsA = torch.stack(torch.meshgrid(torch.arange(A_h), torch.arange(A_w), indexing="ij")).to(GPU)
    patchesA = idxsA[..., None, None] + patch_window[0, :, None, None]
    patchesA = img_A.squeeze()[:, patchesA[0], patchesA[1]].permute(1, 2, 3, 4, 0)
    patchesA = torch.tile(patchesA[:, :, None], [1, 1, K, 1, 1, 1]).reshape(A_h, A_w, K, -1)

    dAB = torch.sum(torch.square(patchesB - patchesA), dim=-1)
    kNNF = torch.stack((dAB, ysB, xsB), dim=-1).cpu()
    for y in torch.arange(A_h):
        for x in torch.arange(A_w):
            kNNF[y, x] = heapify(kNNF[y, x])
    kNNF = kNNF.to(GPU)

    # propagate
    max_side = torch.maximum(torch.tensor(A_h), torch.tensor(A_w))
    ls = torch.floor(max_side / torch.pow(2, torch.arange(torch.log2(max_side))))
    for l in ls:

        kNNF[:, :, 0] = propagate(img_A, img_B, kNNF, l.item(), K, P)

        for y in range(A_h):
            for x in range(A_w):
                siftup(kNNF[y, x], 0)

    return kNNF


if __name__ == "__main__":
    with torch.inference_mode():
        img_A = Image.open("bike_a.png").resize((240, 176))
        img_B = Image.open("bike_b.png").resize((256, 192))

        img_A = to_tensor(img_A).unsqueeze(0).to(GPU)
        img_B = to_tensor(img_B).unsqueeze(0).to(GPU)

        P = 3
        K = 5

        t = time()
        kNNF = patch_match(img_A, img_B, K=K, P=P)
        print(time() - t)

        result = torch.zeros((img_A.shape[2], img_A.shape[3], img_A.shape[1]))
        all_dists = torch.zeros((img_A.shape[2], img_A.shape[3]))
        for y in range(img_A.shape[2]):
            for x in range(img_A.shape[3]):
                all_dists[y, x] = kNNF[y, x, 0, 0]
                result[y, x] = img_B[:, :, kNNF[y, x, 0, 1].long() - P // 2, kNNF[y, x, 0, 2].long() - P // 2].squeeze()
        result = (result.cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(result).save(f"results/bike_{torch.mean(all_dists):.5f}.jpg")
