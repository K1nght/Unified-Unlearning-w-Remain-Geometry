from torch.nn import functional as F
import torch
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

@torch.no_grad()
def entropy(p, dim=-1, keepdim=False):
    return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)

@torch.no_grad()
def m_entropy(p, labels, dim=-1, keepdim=False):
    log_prob = torch.where(p > 0, p.log(), torch.tensor(1e-30).to(p.device).log())
    reverse_prob = 1 - p
    log_reverse_prob = torch.where(
        p > 0, p.log(), torch.tensor(1e-30).to(p.device).log()
    )
    modified_probs = p.clone()
    modified_probs[:, labels] = reverse_prob[:, labels]
    modified_log_probs = log_reverse_prob.clone()
    modified_log_probs[:, labels] = log_prob[:, labels]
    return -torch.sum(modified_probs * modified_log_probs, dim=dim, keepdim=keepdim)

@torch.no_grad()
def collect_prob(data_loader, model):
    model.eval()
    prob = []
    targets = []
    with torch.no_grad():
        for batch in data_loader:
            batch = [tensor.to(next(model.parameters()).device) for tensor in batch]
            data, target = batch
            output = model(data)
            loss = F.cross_entropy(output, target, reduction="none") 
            # print(loss)
            prob.append(F.softmax(output, dim=-1).data)
            targets.append(target)

    return torch.cat(prob), torch.cat(targets)

@torch.no_grad()
def get_membership_attack_data(retain_loader, forget_loader, test_loader, model, metrics="entropy"):
    retain_prob, retain_lables = collect_prob(retain_loader, model)
    forget_prob, forget_lables = collect_prob(forget_loader, model)
    test_prob, test_lables = collect_prob(test_loader, model)
    if metrics == "entropy":
        # print("member mean", entropy(retain_prob).mean(), entropy(forget_prob).mean(), entropy(test_prob).mean())

        X_r = (
            torch.cat([entropy(retain_prob), entropy(test_prob)])
            .cpu()
            .numpy()
            .reshape(-1, 1)
        )
        Y_r = np.concatenate([np.ones(len(retain_prob)), np.zeros(len(test_prob))])

        X_f = entropy(forget_prob).cpu().numpy().reshape(-1, 1)
        Y_f = np.concatenate([np.ones(len(forget_prob))])
    elif metrics == "m_entropy":
        X_r = (
            torch.cat([m_entropy(retain_prob, retain_lables), m_entropy(test_prob, test_lables)])
            .cpu()
            .numpy()
            .reshape(-1, 1)
        )
        Y_r = np.concatenate([np.ones(len(retain_prob)), np.zeros(len(test_prob))])

        X_f = m_entropy(forget_prob, forget_lables).cpu().numpy().reshape(-1, 1)
        Y_f = np.concatenate([np.ones(len(forget_prob))])
    return X_f, Y_f, X_r, Y_r

@torch.no_grad()
def get_membership_attack_prob(retain_loader, forget_loader, test_loader, model, metrics="entropy"):
    X_f, Y_f, X_r, Y_r = get_membership_attack_data(
        retain_loader, forget_loader, test_loader, model, metrics
    )
    clf = LogisticRegression(
        class_weight="balanced", solver="lbfgs", multi_class="multinomial"
    )
    clf.fit(X_r, Y_r)
    results = clf.predict(X_f)
    train_score = clf.score(X_r, Y_r)
    test_score = clf.score(X_f, Y_f)
    print(f"{metrics} MIA train score: {train_score}, test score: {test_score}")
    return results.mean()


