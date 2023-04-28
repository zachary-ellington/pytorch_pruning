import torch
import numpy as np
from torch.nn.utils import prune

class ThresholdPruning(prune.BasePruningMethod):
    """
    This class is a pruning method that prunes values if the absolute value is below a certain threshold
    """
    PRUNING_TYPE = 'unstructured'
    def __init__(self, threshold):
        self.threshold = threshold

    def compute_mask(self, tensor, default_mask):
        return torch.logical_and(torch.nn.functional.threshold(torch.abs(tensor), self.threshold, 0.0, False) > 0, (default_mask == 1)).float()
        # return (torch.abs(tensor) >= self.threshold).float() * default_mask

def global_mag_weight_prune(model, amount):
    """
    !!! Requires prepare_model to be called on model before use !!!
    !!! Requires model to have 'weight' parameter !!!
    Prunes weights whose absolute value are below the specified percentile

    model: a pytorch model
    amount: the specified percentile as a value in 0..1
    """

    # store the parameters that will be pruned for the global unstructured pruning
    parameters = list()

    all_relevant_weights = np.array([])

    # the below for-loop (and its inner for-loop) iterate through all the parameters, checking for 'weight' parameters
    #   and all the values of all the weights are then put inside the array 'all_relevant_weights' which will be used to calculate the appropriate threshold

    for module in model.modules():
        for param_name, _ in module.named_parameters():
            if (prune.is_pruned(module) and param_name == 'weight_orig'):
                parameters.append((module, 'weight'))
                weight_mask = module.get_buffer('weight_mask')
                weight_orig = module.get_parameter('weight_orig')
                relevant = torch.masked_select(weight_orig, weight_mask == 1).flatten().cpu().detach().numpy()
                all_relevant_weights = np.append(all_relevant_weights, relevant)

    threshold = np.percentile(np.abs(all_relevant_weights), amount * 100.0)

    prune.global_unstructured(
        parameters=parameters,
        pruning_method=ThresholdPruning,
        threshold = threshold
    )
