import torch


def cg_batch(A_torch, B, rtol=1e-3, atol=0., maxiter=None):
#A modified code from https://github.com/sbarratt/torch_cg/blob/master/torch_cg/cg_batch.py

    if maxiter is None:
        maxiter = 30


    X0 = B
    X_k = X0
    R_k = B - A_torch @ X_k
    Z_k = R_k

    P_k = torch.zeros_like(Z_k)

    P_k1 = P_k
    R_k1 = R_k
    R_k2 = R_k
    X_k1 = X0
    Z_k1 = Z_k
    Z_k2 = Z_k

    B_norm = torch.norm(B, dim=1)
    stopping_matrix = torch.max(rtol*B_norm, atol*torch.ones_like(B_norm))



    for k in range(1, maxiter + 1):
        Z_k = R_k

        if k == 1:
            P_k = Z_k
            R_k1 = R_k
            X_k1 = X_k
            Z_k1 = Z_k
        else:
            R_k2 = R_k1
            Z_k2 = Z_k1
            P_k1 = P_k
            R_k1 = R_k
            Z_k1 = Z_k
            X_k1 = X_k
            denominator = (R_k2 * Z_k2).sum(1)
            denominator[denominator == 0] = 1e-8
            beta = (R_k1 * Z_k1).sum(1) / denominator
            P_k = Z_k1 + beta.unsqueeze(1) * P_k1

        denominator = (P_k *  (A_torch @ P_k)).sum(1)
        denominator[denominator == 0] = 1e-8
        alpha = (R_k1 * Z_k1).sum(1) / denominator
        X_k = X_k1 + alpha.unsqueeze(1) * P_k
        R_k = R_k1 - alpha.unsqueeze(1) * (A_torch @ P_k)


        residual_norm = torch.norm( (A_torch @ X_k) - B, dim=1)
        
        if (residual_norm <= stopping_matrix).all():
            break
            
    return X_k
