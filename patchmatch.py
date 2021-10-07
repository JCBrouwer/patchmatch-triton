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

import heapq

import numpy as np
from PIL import Image
from tqdm import tqdm

img_A = np.asarray(Image.open("PatchMatch-Pytorch/bike_a.png")) / 255
img_B = np.asarray(Image.open("PatchMatch-Pytorch/bike_b.png").resize((img_A.shape[1] + 25, img_A.shape[0] - 12))) / 255

num_neigh = 7
patch_size = 3
p = patch_size - 1

A_h, A_w = img_A.shape[:2]
B_h, B_w = img_B.shape[:2]

kNNF = [[[] for _ in range(A_w)] for _ in range(A_h)]
hash = [[set() for _ in range(A_w)] for _ in range(A_h)]


def distance(a, b):
    return -np.sum(np.square(b - a))  # negative so that heapq is max heap


print("initializing")
for y in range(A_h):
    for x in range(A_w):
        for k in range(num_neigh):

            idx = np.random.randint(B_h * B_w)

            yB, xB = idx // B_w, idx % B_w

            dAB = distance(img_B[yB, xB], img_A[y, x])

            heapq.heappush(kNNF[y][x], (dAB, (min(B_h - p - 1, yB), min(B_w - p - 1, xB))))

            hash[y][x].add((yB, xB))


print("propagating")
for iter in tqdm(range(5)):

    l = max(img_A.shape[:2])
    while l >= 1:

        for y in tqdm(range(A_h - p)):
            for x in range(A_w - p):

                # TODO where to use hashmap?

                candidates = [kNNF[y][x][0][1]]
                for k in range(num_neigh):
                    for h in [-l, 0, l]:
                        for w in [-l, 0, l]:
                            yC = max(0, min(A_h - 1, y + h))
                            xC = max(0, min(A_w - 1, x + w))
                            candidates.append(kNNF[yC][xC][k][1])
                candidates = np.unique(candidates, axis=0)

                patch = img_A[y : y + patch_size, x : x + patch_size]
                dists = []
                for yC, xC in candidates:
                    dists.append(
                        distance(
                            img_B[yC : yC + patch_size, xC : xC + patch_size],
                            img_A[y : y + patch_size, x : x + patch_size],
                        )
                    )
                best = np.argmax(dists)  # distances are negative for heapq => argmax gives min dist

                heapq.heapreplace(kNNF[y][x], (dists[best], tuple(candidates[best])))

        temp = np.zeros_like(img_A)
        dists = []
        for i in range(A_h):
            for j in range(A_w):
                best_patch = kNNF[i][j][0]
                dists.append(best_patch[0])
                temp[i, j, :] = img_B[best_patch[1][0], best_patch[1][1]]
        Image.fromarray((temp * 255).astype(np.uint8)).save(f"{iter}_{l}_{-np.mean(dists):.5f}.jpg")

        l //= 2
