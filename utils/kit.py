
import torch
import math
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from torch.distributions import Beta

def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def getMaxValue(mylist):

    if mylist == None:
        return 0
    elif len(mylist) == 1:
        return mylist[0]


    elif mylist[0] > getMaxValue(mylist[1:]):
        return mylist[0]
    else:
        return getMaxValue(mylist[1:])

def replace_inf_to_zero(val):
    val[val == float('inf')] = 0.0
    return val

def entropy_loss(mask, logits_s, prob_model, label_hist):

    mask = mask.bool()

    # select samples
    logits_s = logits_s[mask]

    prob_s = logits_s.softmax(dim=-1)
    _, pred_label_s = torch.max(prob_s, dim=-1)

    hist_s = torch.bincount(pred_label_s, minlength=logits_s.shape[1]).to(logits_s.dtype)
    hist_s = hist_s / hist_s.sum()

    # modulate prob model
    prob_model = prob_model.reshape(1, -1)
    label_hist = label_hist.reshape(1, -1)
    # prob_model_scaler = torch.nan_to_num(1 / label_hist, nan=0.0, posinf=0.0, neginf=0.0).detach()
    prob_model_scaler = replace_inf_to_zero(1 / label_hist).detach()
    mod_prob_model = prob_model * prob_model_scaler
    mod_prob_model = mod_prob_model / mod_prob_model.sum(dim=-1, keepdim=True)

    # modulate mean prob
    mean_prob_scaler_s = replace_inf_to_zero(1 / hist_s).detach()
    # mean_prob_scaler_s = torch.nan_to_num(1 / hist_s, nan=0.0, posinf=0.0, neginf=0.0).detach()
    mod_mean_prob_s = prob_s.mean(dim=0, keepdim=True) * mean_prob_scaler_s
    mod_mean_prob_s = mod_mean_prob_s / mod_mean_prob_s.sum(dim=-1, keepdim=True)

    loss = mod_prob_model * torch.log(mod_mean_prob_s + 1e-12)
    loss = loss.sum(dim=1)
    return loss.mean(), hist_s.mean()

def convert(x):
    x = x.reshape(x.size(0), x.size(1) * x.size(2), x.size(3), x.size(4))
    return x

def convert2(x):
    x1 = x.size(1)
    x2 = x.size(2)
    x = x.reshape(x.size(0), x.size(1) * x.size(2), x.size(3), x.size(4))
    return x, x1, x2


def deconvert2(x,x1,x2):
    x = x.reshape(x.size(0), x1, x2, x.size(-2), x.size(-1))
    return x

def KL_div(prob, target, reduction):
    eps = 1e-16
    b, c, *hwd = target.shape
    kl = (-target * torch.log((prob + eps) / (target + eps)))
    if reduction == "mean":
        return kl.mean()
    elif reduction == "sum":
        return kl.sum()

#Original Mixup
def mixup(label_img: torch.Tensor, label_onehot: torch.Tensor, unlab_img: torch.Tensor,
          unlabeled_pred: torch.Tensor, input_lablel_num, args):


    labeled_idx =  label_img.shape[0]
    unlabeled_idx = unlab_img.shape[0]

    if labeled_idx > unlabeled_idx:
        front_img = label_img[:input_lablel_num]
        front_onehot = label_onehot[:input_lablel_num]
        random_indices = torch.randperm(labeled_idx-input_lablel_num)[:unlabeled_idx-input_lablel_num]
        random_img = label_img[input_lablel_num:][random_indices]
        random_onehot = label_onehot[input_lablel_num:][random_indices]
        label_img = torch.cat((front_img,random_img))
        label_onehot = torch.cat((front_onehot,random_onehot))

    if labeled_idx <= unlabeled_idx:
        idx = unlabeled_idx / labeled_idx
        idx = int(idx)

        res = unlabeled_idx - (idx * labeled_idx)
        #res_img = label_img[0:res, :, :, :]
        #res_onehot = label_onehot[0:res, :]
        #if res==0:
            #label_onehot = torch.unsqueeze(label_onehot, dim=0)
        res_img = label_img[0:res, :, :, :]
        res_onehot = label_onehot[0:res, :]


        l_i = label_img
        l_o = label_onehot
        for i in range(idx - 1):
            label_img = torch.cat([label_img, l_i], 0)
            label_onehot = torch.cat([label_onehot, l_o], 0)
        if res != 0:
            label_img = torch.cat([label_img, res_img], 0)
            label_onehot = torch.cat([label_onehot, res_onehot], 0)
            
            

    num_samples = label_img.size(0)
    random_indices = np.random.permutation(num_samples)
    label_img = label_img[random_indices]
    label_onehot = label_onehot[random_indices]

    assert label_img.shape == unlab_img.shape
    assert label_img.shape.__len__() == 4
    # assert F.one_hot(label_onehot) and simplex(unlabeled_pred)
    assert label_onehot.shape == unlabeled_pred.shape


    beta_distr: Beta = Beta(torch.tensor([1.0]), torch.tensor([1.0]))
    device = 'cuda'

    bn, *shape = label_img.shape
    alpha = beta_distr.sample((bn,)).squeeze(1).to(args.device)
    _alpha = alpha.view(bn, 1, 1, 1).repeat(1, *shape)
    assert _alpha.shape == label_img.shape
    mixup_img = label_img * _alpha + unlab_img * (1 - _alpha)
    mixup_label = label_onehot * alpha.view(bn, 1) \
                  + unlabeled_pred * (1 - alpha).view(bn, 1)
    mixup_index = torch.stack([alpha, 1 - alpha], dim=1).to(args.device)

    assert mixup_img.shape == label_img.shape
    assert mixup_label.shape == label_onehot.shape
    assert mixup_index.shape[0] == bn
    # assert simplex(mixup_index)

    return mixup_img, mixup_label, mixup_index
    
def mixup1(label_img: torch.Tensor, label_onehot: torch.Tensor, unlab_img: torch.Tensor,
          unlabeled_pred: torch.Tensor, input_lablel_num, args):


    labeled_idx =  label_img.shape[0]
    unlabeled_idx = unlab_img.shape[0]

    if labeled_idx > unlabeled_idx:
        front_img = label_img[:input_lablel_num]
        front_onehot = label_onehot[:input_lablel_num]
        random_indices = torch.randperm(labeled_idx-input_lablel_num)[:unlabeled_idx-input_lablel_num]
        random_img = label_img[input_lablel_num:][random_indices]
        random_onehot = label_onehot[input_lablel_num:][random_indices]
        label_img = torch.cat((front_img,random_img))
        label_onehot = torch.cat((front_onehot,random_onehot))

    if labeled_idx <= unlabeled_idx:
        idx = unlabeled_idx / labeled_idx
        idx = int(idx)

        res = unlabeled_idx - (idx * labeled_idx)
        #res_img = label_img[0:res, :, :, :]
        #res_onehot = label_onehot[0:res, :]
        #if res==0:
            #label_onehot = torch.unsqueeze(label_onehot, dim=0)
        res_img = label_img[0:res, :, :, :]
        if label_onehot.size(0) != 1:
            res_onehot = label_onehot.unsqueeze(dim=0)


        l_i = label_img
        l_o = label_onehot
        for i in range(idx - 1):
            label_img = torch.cat([label_img, l_i], 0)
            label_onehot = torch.cat([label_onehot, l_o], 0)
        if res != 0:
            label_img = torch.cat([label_img, res_img], 0)
            label_onehot = torch.cat([label_onehot, res_onehot], 0)

    num_samples = label_img.size(0)
    random_indices = np.random.permutation(num_samples)
    label_img = label_img[random_indices]
    label_onehot = label_onehot.unsqueeze(dim=0)

    assert label_img.shape == unlab_img.shape
    assert label_img.shape.__len__() == 4
    # assert F.one_hot(label_onehot) and simplex(unlabeled_pred)
    assert label_onehot.shape == unlabeled_pred.shape


    beta_distr: Beta = Beta(torch.tensor([1.0]), torch.tensor([1.0]))
    device = 'cuda'

    bn, *shape = label_img.shape
    alpha = beta_distr.sample((bn,)).squeeze(1).to(args.device)
    _alpha = alpha.view(bn, 1, 1, 1).repeat(1, *shape)
    assert _alpha.shape == label_img.shape
    mixup_img = label_img * _alpha + unlab_img * (1 - _alpha)
    mixup_label = label_onehot * alpha.view(bn, 1) \
                  + unlabeled_pred * (1 - alpha).view(bn, 1)
    mixup_index = torch.stack([alpha, 1 - alpha], dim=1).to(args.device)

    assert mixup_img.shape == label_img.shape
    assert mixup_label.shape == label_onehot.shape
    assert mixup_index.shape[0] == bn
    # assert simplex(mixup_index)

    return mixup_img, mixup_label, mixup_index