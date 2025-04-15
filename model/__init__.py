import torch
import torch.nn as nn


class ExpLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, input):
        return nn.functional.linear(input, self.weight.exp(), self.bias.exp())


class ScoringModel(torch.nn.Module):
    def __init__(self, num_of_rules: int) -> None:
        assert num_of_rules > 0

        super(ScoringModel, self).__init__()
        self.raw_rule_scores = torch.nn.Parameter(
            torch.randn((num_of_rules, 1), dtype=torch.float32)
        )
        self.pos_rule_scores = torch.nn.Softplus()

    def forward(self, confidence):
        score = torch.matmul(
            confidence, self.pos_rule_scores(self.raw_rule_scores)
        )
        total_score = torch.sum(self.pos_rule_scores(self.raw_rule_scores))
        normalized_score = score / total_score
        return normalized_score