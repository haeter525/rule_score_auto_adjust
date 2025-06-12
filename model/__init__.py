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


class ScoringModel(nn.Module):
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

    def get_rule_scores(self):
        return self.pos_rule_scores(self.raw_rule_scores)


class RuleAdjustmentModel(nn.Module):
    def __init__(self, num_of_rules: int) -> None:
        """Initialize the Rule Adjustment Model."""
        super(RuleAdjustmentModel, self).__init__()

        # Initialize the rule scores with random values to allow for learning
        self.rule_score = torch.nn.Parameter(
            torch.randn((num_of_rules,), dtype=torch.float32)
        )
        self.normalize = torch.nn.Sigmoid()

    def __convert_to_weights(self, stage: torch.Tensor) -> torch.Tensor:
        """Convert the stage to a weight for the rule score."""
        numerator = torch.scalar_tensor(2, dtype=torch.float32) ** stage
        denominator = torch.scalar_tensor(2**5, dtype=torch.float32)
        return numerator / denominator
    
    def calculate_apk_scores(
        self, passing_stages: torch.Tensor
    ) -> torch.Tensor:
        # score_weights = passing_stages.type(torch.float32).apply_(
        #     self.__convert_to_weights
        # )
        score_weights = passing_stages

        apk_scores = torch.matmul(score_weights, self.rule_score)
        return apk_scores 

    def forward(self, passing_stages: torch.Tensor) -> torch.Tensor:
        """The main logic of the model."""
        apk_scores = self.calculate_apk_scores(passing_stages)
        total_score = self.rule_score.sum()
        return self.normalize((apk_scores / total_score) - 0.5)

    def get_rule_scores(self):
        return self.rule_score


class RuleAdjustmentModel_NoTotalScore(nn.Module):
    def __init__(self, num_of_rules: int) -> None:
        """Initialize the Rule Adjustment Model."""
        super(RuleAdjustmentModel_NoTotalScore, self).__init__()

        # Initialize the rule scores with random values to allow for learning
        self.rule_score = torch.nn.Parameter(
            torch.randn((num_of_rules,), dtype=torch.float32)
        )

        # Normalize the result to be between 0 and 1
        self.normalize = torch.nn.Sigmoid()

    def forward(self, passing_stages: torch.Tensor) -> torch.Tensor:
        """The main logic of the model."""
        # score_weights = passing_stages.type(torch.float32).apply_(
        #     self.__convert_to_weights
        # )
        score_weights = passing_stages

        apk_scores = torch.matmul(score_weights, self.rule_score)
        classification = self.normalize(apk_scores)
        return classification

    def get_rule_scores(self):
        return self.rule_score


class RuleAdjustmentModel_NoTotalScore_Percentage(nn.Module):
    def __init__(self, num_of_rules: int) -> None:
        """Initialize the Rule Adjustment Model."""
        super(RuleAdjustmentModel_NoTotalScore_Percentage, self).__init__()

        # Initialize the rule scores with random values to allow for learning
        self.rule_score = torch.nn.Parameter(
            torch.randn((num_of_rules,), dtype=torch.float32)
        )

        # Normalize the result to be between 0 and 1
        self.normalize = torch.nn.Sigmoid()

    def calculate_apk_scores(
        self, passing_stages: torch.Tensor
    ) -> torch.Tensor:
        # score_weights = passing_stages.type(torch.float32).apply_(
        #     self.__convert_to_weights
        # )
        score_weights = passing_stages

        apk_scores = torch.matmul(score_weights, self.rule_score)
        apk_scores_percent = apk_scores / len(self.rule_score)
        return apk_scores_percent

    def forward(self, passing_stages: torch.Tensor) -> torch.Tensor:
        """The main logic of the model."""
        apk_scores_percent = self.calculate_apk_scores(passing_stages)
        classification = self.normalize(apk_scores_percent)
        return classification

    def get_rule_scores(self):
        return self.rule_score
