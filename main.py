#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import pickle
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
from scipy.io import loadmat
from joblib import Parallel, delayed
from tqdm import tqdm
import torch
import torch.nn.functional as F

from knn import KNN, ANN
from cg_batch import cg_batch
from revisitop.evaluate import compute_map


def load_features(mat_path):
    features = loadmat(mat_path)
    queries = features["Q"].transpose()
    gallery = features["X"].transpose()
    return queries, gallery


def load_gnd(gnd_path):
    with open(gnd_path, "rb") as f:
        cfg = pickle.load(f)
    return cfg["gnd"]


def build_knn_index(features):
    use_ann = len(features) >= 100000
    if use_ann:
        return ANN(features, method="cosine")
    return KNN(features, method="cosine")


def get_laplacian(affinity, alpha=0.99):
    num = affinity.shape[0]
    degrees = affinity @ np.ones(num) + 1e-12
    mat = sparse.dia_matrix((degrees ** (-0.5), [0]), shape=(num, num), dtype=np.float32)
    stochastic = mat @ affinity @ mat
    sparse_eye = sparse.dia_matrix((np.ones(num), [0]), shape=(num, num), dtype=np.float32)
    return sparse_eye - alpha * stochastic


def get_affinity(sims, ids, gamma=3, flag=1):
    num = sims.shape[0]
    sims = sims.copy()
    sims[sims < 0] = 0
    sims = sims ** gamma
    vec_ids, mut_ids, mut_sims = [], [], []
    for i in range(num):
        if flag == 1:
            vec_ids.append(i * np.ones(ids.shape[1]))
            mut_ids.append(ids[i, :])
            mut_sims.append(sims[i, :])
        elif flag == 2:
            ismutual = np.isin(ids[ids[i]], i).any(axis=1)
            if ismutual.any():
                vec_ids.append(i * np.ones(ismutual.sum()))
                mut_ids.append(ids[i, ismutual])
                opposite_sims = sims[ids[i, ismutual],
                                    np.where(ids[ids[i, ismutual]] == i)[1]]
                mut_sims.append(np.minimum(sims[i, ismutual], opposite_sims))
    if not vec_ids:
        return sparse.csc_matrix((num, num), dtype=np.float32)
    vec_ids, mut_ids, mut_sims = map(np.concatenate, [vec_ids, mut_ids, mut_sims])
    vec_ids = vec_ids.astype(np.int32)
    mut_ids = mut_ids.astype(np.int32)
    affinity = sparse.csc_matrix((mut_sims, (vec_ids, mut_ids)),
                                 shape=(num, num), dtype=np.float32)
    affinity[range(num), range(num)] = 0
    return affinity


def get_offline_result(i, trunc_ids, trunc_init, lap_alpha):
    ids = trunc_ids[i]
    trunc_lap = lap_alpha[ids][:, ids]
    scores, _ = linalg.cg(trunc_lap, trunc_init, rtol=1e-6, maxiter=20)
    ranks = np.argsort(-scores)
    return scores[ranks], ids[ranks]


def build_offline_matrix(lap_alpha, trunc_ids, n_trunc):
    trunc_init = np.zeros(n_trunc, dtype=np.float32)
    trunc_init[0] = 1
    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(get_offline_result)(i, trunc_ids, trunc_init, lap_alpha)
        for i in tqdm(range(trunc_ids.shape[0]), desc="[offline] diffusion")
    )
    all_scores, all_ranks = map(np.concatenate, zip(*results))
    rows = np.repeat(np.arange(trunc_ids.shape[0]), n_trunc)
    return sparse.csr_matrix((all_scores, (rows, all_ranks)),
                             shape=(trunc_ids.shape[0], trunc_ids.shape[0]),
                             dtype=np.float32)


def query_offline(offline, knn_index, queries, kq, gamma, n_trunc):
    time0 = time.time()
    print("[search] 1) k-NN search")
    sims, ids = knn_index.search(queries, kq)
    sims = sims ** gamma
    qr_num = ids.shape[0]
    print("[search] 2) linear combination")
    all_ranks = np.empty((qr_num, n_trunc), dtype=int)
    for i in tqdm(range(qr_num), desc="[search] query"):
        scores = sims[i] @ offline[ids[i]]
        scores = np.asarray(scores).ravel()
        if n_trunc == scores.shape[0]:
            parts = np.arange(scores.shape[0])
        else:
            parts = np.argpartition(-scores, n_trunc)[:n_trunc]
        ranks = np.argsort(-scores[parts])
        all_ranks[i] = parts[ranks]
    print("[search] search costs {:.2f}s".format(time.time() - time0))
    return all_ranks.T


def gaussian(x, std):
    sig2 = 2 * std * std
    return torch.exp(-x ** 2 / sig2)


def crf_denoise(local_sims, local_ids, gallery, batch_size=15, use_gpu=True, gpu_id=0,
                alpha=1.0, beta=0.1, g_weight=0.9, kl_weight=0.00035):
    device = torch.device("cpu")
    if use_gpu and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")

    t_sims = torch.tensor(local_sims, device=device)
    t_ids = torch.tensor(local_ids, device=device, dtype=torch.long)
    gallery_t = torch.tensor(gallery, device=device)

    probescore = t_sims
    mu = probescore.clone()
    ind_list = np.arange(probescore.shape[0])
    diag_idx = torch.arange(t_ids.shape[1], device=device)

    for p in tqdm(range(0, probescore.shape[0], batch_size), desc="[c-crf] denoise"):
        batches = ind_list[p: p + batch_size]
        batches_t = torch.tensor(batches, device=device, dtype=torch.long)

        galleries_score = torch.bmm(
            gallery_t[t_ids[batches_t]],
            gallery_t[t_ids[batches_t]].transpose(2, 1)
        )
        galleries_score[galleries_score < 0] = 0

        pairwise_mat = torch.sqrt(F.relu(2 - 2 * galleries_score))
        normed_pairwise = F.normalize(galleries_score, dim=2, p=2)
        softmax_prob = F.softmax(normed_pairwise, dim=2)

        temp = torch.bmm(softmax_prob, torch.log(softmax_prob).transpose(2, 1))
        temp_diag = temp[:, diag_idx, diag_idx].reshape(temp.shape[0], -1, 1)
        kldiv_mat = (temp_diag - temp)
        jsdiv_mat = (kldiv_mat + kldiv_mat.transpose(2, 1)) / 2
        weights = gaussian(pairwise_mat, g_weight) * gaussian(jsdiv_mat, kl_weight)

        weights[:, diag_idx, diag_idx] = 0
        A_torch = beta * weights.sum(2).diag_embed() - beta * weights
        A_torch[:, diag_idx, diag_idx] += alpha

        B_torch = alpha * probescore[batches_t].reshape(A_torch.shape[0], -1, 1)
        mu[batches_t] = cg_batch(A_torch, B_torch, rtol=1e-5, atol=1e-5, maxiter=20).squeeze()

    mu = mu.cpu().numpy()
    order = np.argsort(-mu, axis=1)
    ids = np.take_along_axis(local_ids, order, axis=1)
    sims = np.take_along_axis(mu, order, axis=1)
    return sims, ids


def evaluate_ranks(ranks, gnd, test_dataset):
    def build_gnd(mode):
        gnd_t = []
        for i in range(len(gnd)):
            if mode == "easy":
                ok = np.concatenate([gnd[i]["easy"]])
                junk = np.concatenate([gnd[i]["junk"], gnd[i]["hard"]])
            elif mode == "medium":
                ok = np.concatenate([gnd[i]["easy"], gnd[i]["hard"]])
                junk = np.concatenate([gnd[i]["junk"]])
            else:
                ok = np.concatenate([gnd[i]["hard"]])
                junk = np.concatenate([gnd[i]["junk"], gnd[i]["easy"]])
            gnd_t.append({"ok": ok, "junk": junk})
        return gnd_t

    mapE, _, _, _ = compute_map(ranks, build_gnd("easy"))
    mapM, _, _, _ = compute_map(ranks, build_gnd("medium"))
    mapH, _, _, _ = compute_map(ranks, build_gnd("hard"))
    print(">> {}: mAP E: {}, M: {}, H: {}".format(
        test_dataset,
        np.around(mapE * 100, decimals=2),
        np.around(mapM * 100, decimals=2),
        np.around(mapH * 100, decimals=2),
    ))


def run_ours_c_crf(queries, gallery, knn_index, full_sims, full_ids,
                   gamma, n_trunc, kd, kq, clique_size):
    print("[offline] starting offline diffusion (ours)")
    local_ids = full_ids[:, :clique_size]
    local_sims = full_sims[:, :clique_size]
    sims, ids = crf_denoise(local_sims, local_ids, gallery)

    trunc_ids = full_ids[:, :n_trunc]
    affinity = get_affinity(sims[:, :kd], ids[:, :kd], gamma=gamma, flag=1)
    affinity = (affinity + affinity.transpose()) / 2
    lap_alpha = get_laplacian(affinity)

    offline = build_offline_matrix(lap_alpha, trunc_ids, n_trunc)
    return query_offline(offline, knn_index, queries, kq, gamma, n_trunc)


def run_offline_diffusion(queries, knn_index, full_sims, full_ids,
                          gamma, n_trunc, kd, kq):
    print("[offline] starting offline diffusion (baseline)")
    trunc_ids = full_ids[:, :n_trunc]
    affinity = get_affinity(full_sims[:, :kd], full_ids[:, :kd], gamma=gamma, flag=2)
    lap_alpha = get_laplacian(affinity)

    offline = build_offline_matrix(lap_alpha, trunc_ids, n_trunc)
    return query_offline(offline, knn_index, queries, kq, gamma, n_trunc)


def run_knn_baseline(queries, gallery, n_trunc):
    sim = np.dot(gallery, queries.T)
    return np.argsort(-sim, axis=0)[:n_trunc]


def main():
    feature_path = "revisitop/data/roxford5k_resnet_rsfm120k_gem.mat"
    gnd_path = "revisitop/data/gnd_roxford5k.pkl"
    test_dataset = "ROxford5k"

    gamma = 3
    n_trunc = 1000
    kd = 50
    kq = 10
    clique_size = 1000

    queries, gallery = load_features(feature_path)
    gnd = load_gnd(gnd_path)

    knn_index = build_knn_index(gallery)
    full_sims, full_ids = knn_index.search(gallery, gallery.shape[0])

    print("\n== 1) ours (c-crf) ==")
    ranks = run_ours_c_crf(queries, gallery, knn_index, full_sims, full_ids,
                           gamma, n_trunc, kd, kq, clique_size)
    evaluate_ranks(ranks, gnd, test_dataset)

    print("\n== 2) offline diffusion ==")
    ranks = run_offline_diffusion(queries, knn_index, full_sims, full_ids,
                                  gamma, n_trunc, kd, kq)
    evaluate_ranks(ranks, gnd, test_dataset)

    print("\n== 3) k-NN search ==")
    ranks = run_knn_baseline(queries, gallery, n_trunc)
    evaluate_ranks(ranks, gnd, test_dataset)


if __name__ == "__main__":
    main()
