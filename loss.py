import torch.nn.functional as F
import torch.nn as nn
import torch

class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, smoothing=0.1, temperature=0.5):
        super().__init__()
        self.smoothing = smoothing
        self.temperature = temperature

    def forward(self, x, target):
        log_probs = F.log_softmax(x / self.temperature, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(dim=-1)).squeeze(dim=-1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    @staticmethod
    def get_anchor_positive_triplet_mask(target):
        mask = torch.eq(target.unsqueeze(0), target.unsqueeze(1))
        mask = mask * (~ torch.diag(torch.ones(target.size()))).float()
        return mask

    @staticmethod
    def get_anchor_negative_triplet_mask(target):
        labels_equal = torch.eq(target.unsqueeze(0), target.unsqueeze(1))
        mask = ~ labels_equal
        return mask

    def forward(self, x, target):
        pairwise_dist = torch.pairwise_distance(x.unsqueeze(0), x.unsqueeze(1), p=2)
        mask_anchor_positive = self.get_anchor_positive_triplet_mask(target)
        anchor_positive_dist = mask_anchor_positive * pairwise_dist
        hardest_positive_dist = anchor_positive_dist.max(1, True)[0]

        mask_anchor_negative = self.get_anchor_negative_triplet_mask(target)
        # make positive and anchor to be exclusive through maximizing the dist
        max_anchor_negative_dist = pairwise_dist.max(1, True)[0]
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative.float())
        hardest_negative_dist = anchor_negative_dist.min(1, True)[0]

        loss = (F.relu(hardest_positive_dist - hardest_negative_dist + self.margin))
        return loss.mean()


def compute_feature_similarity_loss(feature_emdeddings, target):
    loss = torch.Tensor([0.0]).cuda()
    if len(feature_emdeddings) == 1:
        pass
    else:
        for i in range(1, len(feature_emdeddings)-1):
            loss += F.margin_ranking_loss(feature_emdeddings[i], feature_emdeddings[i+1], target)
    return loss


def triplet_loss(inputs, targets, margin=0.3):
    n = inputs.size(0)
    ranking_loss = nn.MarginRankingLoss(margin=margin)

    dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist = dist + dist.t()
    dist.addmm_(1, -2, inputs, inputs.t())
    dist = dist.clamp(min=1e-12).sqrt()

    mask = targets.expand(n, n).eq(targets.expand(n, n).t())
    dist_ap, dist_an = [], []
    for i in range(n):
        dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
        dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
    dist_ap = torch.cat(dist_ap)
    dist_an = torch.cat(dist_an)

    y = torch.ones_like(dist_an)
    loss = ranking_loss(dist_an, dist_ap, y)
    return loss


if __name__ == "__main__":
    input1_1 = torch.Tensor([[1, 3]])
    input2_1 = torch.Tensor([[1, 3]])

