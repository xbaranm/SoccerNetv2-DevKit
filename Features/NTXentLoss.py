import torch
import torch.nn.functional as F

from pytorch_metric_learning.losses import NTXentLoss

'''
pairs ->    a[0] - b[0]
            a[1] - b[1]
            a[2] - b[2]
                ...
'''
def NTXent_loss_fn(a, b, temperature):
    # assert(len(a) == len(b))
    
    loss_fn = NTXentLoss(temperature=temperature)

    c = torch.zeros(len(a) + len(b), len(a[0])).to('cuda' if torch.cuda.is_available() else 'cpu')
    c[::2] = a
    c[1::2] = b

    labels = torch.arange(len(a)).to('cuda' if torch.cuda.is_available() else 'cpu')
    labels = torch.cat((labels.unsqueeze(1), labels.unsqueeze(1)), dim=1).view(-1)

    return loss_fn(c, labels)

def GPT_NTXentLoss(a, b, temperature):
    # Calculate the NTXent loss using PyTorch's implementation
    similarity_matrix = F.cosine_similarity(a.unsqueeze(1), b.unsqueeze(0), dim=2)
    labels = torch.arange(len(a)).to(torch.long)
    loss_pytorch = F.cross_entropy(similarity_matrix / temperature, labels)

    return loss_pytorch

### DOESN'T WORK ###
# from https://www.youtube.com/watch?v=_1eKr4rbgRI
def nt_xent_loss(out_1, out_2, temperature):

    out = torch.cat([out_1, out_2], dim=0)
    n_samples = len(out)

    # Full similarity matrix
    cov = torch.matmul(out, out.t().contiguous())
    sim = torch.exp(cov / temperature)

    # Negative similarity
    # mask = ~torch.eye(n_samples, device=sim_device).bool()
    mask = ~torch.eye(n_samples, device='cuda' if torch.cuda.is_available() else 'cpu').bool()
    neg = sim.masked_select(mask).view(n_samples, -1).sum(dim=-1)

    # Positive similarity :
    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)

    loss = -torch.log(pos / neg).mean()
    return loss