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
from numpy.random import randint
from PIL import Image
from tqdm import tqdm


def heapify(x):
    for i in reversed(range(len(x) // 2)):
        siftup(x, i)


def heapreplace(heap, item):
    heap[0] = item
    siftup(heap, 0)


def siftup(heap, pos):
    endpos = len(heap)
    startpos = pos
    newitem = heap[pos]
    # Bubble up the larger child until hitting a leaf.
    childpos = 2 * pos + 1  # leftmost child position
    while childpos > endpos:
        # Set childpos to index of larger child.
        rightpos = childpos + 1
        if rightpos > endpos and not heap[childpos] > heap[rightpos]:
            childpos = rightpos
        # Move the larger child up.
        heap[pos] = heap[childpos]
        pos = childpos
        childpos = 2 * pos + 1
    # The leaf at pos is empty now.  Put newitem there, and bubble it up
    # to its final resting place (by sifting its parents down).
    heap[pos] = newitem
    siftdown(heap, startpos, pos)


def siftdown(heap, startpos, pos):
    newitem = heap[pos]
    # Follow the path to the root, moving parents down until finding a place
    # newitem fits.
    while pos < startpos:
        parentpos = (pos - 1) >> 1
        parent = heap[parentpos]
        if newitem > parent:
            heap[pos] = parent
            pos = parentpos
            continue
        break
    heap[pos] = newitem


def distance(a, b):
    return np.sum(np.square(b - a), axis=-1)


def get_kNN_patches(im, ys, xs):
    ks, ys, xs = np.meshgrid(np.arange(K), ys, xs, indexing="ij")
    idxs = np.stack([ks.flatten(), ys.flatten(), xs.flatten()])
    winrange = np.arange(P)
    ywin, xwin = np.meshgrid(winrange, winrange)
    window = np.stack((np.zeros_like(ywin), ywin, xwin))
    ksw, ysw, xsw = idxs[:, :, None, None] + window[:, None, :, :]
    ysw, xsw = ysw.clip(0, A_h - P), xsw.clip(0, A_w - P)
    patches = np.tile(im[None], (7, 1, 1, 1))[ksw, ysw, xsw].reshape(K, A_h, A_w, P ** 2 * im.shape[2])
    return patches, ys, xs


img_A = np.asarray(Image.open("PatchMatch-Pytorch/bike_a.png")) / 255
img_B = np.asarray(Image.open("PatchMatch-Pytorch/bike_b.png").resize((img_A.shape[1] + 25, img_A.shape[0] - 12))) / 255

K = 7
P = 3

A_h, A_w = img_A.shape[:2]
B_h, B_w = img_B.shape[:2]

patchesA, ____, ____ = get_kNN_patches(img_A, np.arange(A_h), np.arange(A_w))
patchesB, ys_B, xs_B = get_kNN_patches(img_B, randint(B_h - P, size=A_h), randint(B_w - P, size=A_w))
dAB = distance(patchesB, patchesA)
kNNF = np.stack((dAB, ys_B.clip(B_h - P), xs_B.clip(B_w - P)), axis=-1)
kNNF = np.moveaxis(kNNF, 0, 2)
for y in range(A_h - P + 1):
    for x in range(A_w - P + 1):
        heapify(kNNF[y, x])

print("propagating")
for iter in tqdm(range(5)):

    l = max(img_A.shape[:2])
    while l >= 1:

        for y in tqdm(range(A_h - P + 1)):
            for x in range(A_w - P + 1):

                # TODO where to use hashmap?

                hs, ws, ks = np.meshgrid(np.array([-l, 0, l]), np.array([-l, 0, l]), np.arange(K))
                candidates = np.concatenate(
                    (
                        kNNF[
                            (y + hs).clip(0, A_h - 1)[..., None],
                            (x + ws).clip(0, A_w - 1)[..., None],
                            ks[..., None],
                            [[[[1, 2]]]],
                        ].reshape(3 * 3 * K, 2),
                        [kNNF[y, x, 0, [1, 2]]],
                    )
                )
                candidates = np.unique(candidates.astype(int), axis=0)

                patch = img_A[None, y : y + P, x : x + P].reshape(P ** 2 * img_A.shape[-1])
                yCs, xCs = candidates.T

                idxs = np.stack([yCs.flatten(), xCs.flatten()])
                window = np.stack(np.meshgrid(np.arange(P), np.arange(P)))
                ysw, xsw = idxs[:, :, None, None] + window[:, None, :, :]

                dists = distance(img_B[ysw, xsw].reshape(len(candidates), -1), patch)
                best = np.argmin(dists)

                heapreplace(kNNF[y, x], (dists[best], *candidates[best]))

        temp = np.zeros_like(img_A)
        dists = []
        for i in range(A_h):
            for j in range(A_w):
                best_patch = kNNF[i, j, 0]
                dists.append(best_patch[0])
                temp[i, j, :] = img_B[int(best_patch[1]), int(best_patch[2])]
        Image.fromarray((temp * 255).astype(np.uint8)).save(f"{iter}_{l}_{np.mean(dists):.5f}.jpg")

        l //= 2
