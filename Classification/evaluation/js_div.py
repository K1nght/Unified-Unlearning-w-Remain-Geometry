import torch
import torch.nn.functional as F


def kl_divergence(p, q):
    p_log = torch.log(p + 1e-20)  
    q_log = torch.log(q + 1e-20) 
    kl_div = torch.sum(p * (p_log - q_log), dim=1)
    return kl_div

def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

@torch.no_grad()
def get_js_divergence(forget_loader, unlearned_model, retrained_model):
    unlearn_preds = []
    retrain_preds = []
    for i, (images, labels) in enumerate(forget_loader):
        unlearned_model.eval()
        retrained_model.eval()

        images = images.cuda()
        labels = labels.cuda()
        unlearn_preds.append(F.softmax(unlearned_model(images), dim=1))
        retrain_preds.append(F.softmax(retrained_model(images), dim=1))
    unlearn_preds = torch.cat(unlearn_preds, axis=0)
    retrain_preds = torch.cat(retrain_preds, axis=0)

    return js_divergence(retrain_preds, unlearn_preds).mean().item(), kl_divergence(retrain_preds, unlearn_preds).mean().item()