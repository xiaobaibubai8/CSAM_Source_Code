import torch
import torch.nn.functional as F


class CosineLoss(torch.nn.Module):

    """
    cosine loss:
    https://github.com/novoselrok/codesnippetsearch/blob/11310a8bfc9553df86dd98b120306159fd030b28/code_search/train.py#L95
    """

    def __init__(self):
        super(CosineLoss, self).__init__()

    def forward(self, cosine_similarity_matrix, device: torch.device):
        neg_matrix = torch.zeros(*list(cosine_similarity_matrix.shape)).to(device)
        neg_matrix.fill_diagonal_(float('-inf'))

        # Distance between query and code snippet should be as small as possible
        diagonal_cosine_distance = 1. - torch.diag(cosine_similarity_matrix)
        # Max. similarity between query and non-corresponding code snippet should be as small as possible
        max_positive_non_diagonal_similarity_in_row, _ = torch.max(F.relu(cosine_similarity_matrix + neg_matrix), dim=1)
        # Combined distance and similarity should be as small as possible as well
        per_sample_loss = F.relu(diagonal_cosine_distance + max_positive_non_diagonal_similarity_in_row)

        return torch.mean(per_sample_loss)
