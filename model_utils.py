import torch
from torch.nn.utils import prune
def prepare_model(model):
    # This function is required for several other functions, including pruning
    # This identity prunes all parameters and returns the names of the parameters for reference

    param_names = set()

    global_params = []

    for module in model.modules():
        for param_name, _ in module.named_parameters(recurse = False):
            param_names.add(param_name)
            global_params.append((module, param_name))

    prune.global_unstructured(
        parameters=global_params,
        pruning_method=prune.Identity
    )

    return list(param_names)
            


def get_accuracy(model, dataloader, device):
    # Gets the accuracy of the model when testing the data from the dataloader
    
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, prediction = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (prediction == labels).sum().item()
    return correct / total


def pruned_percentage(model, param_name):
    # Calculates the percentage of the model that is pruned for the parameter name
    # requires prepare_model

    pruned = 0.0
    total = 0.0
    for module in model.modules():
        try:
            mask = module.get_buffer(param_name + "_mask")
            pruned += float(torch.sum(mask == 0))
            total += float(torch.numel(mask))
        except:
            None
    return pruned / total


def remove_pruning(model):
    # Requires prepare_model
    for module in model.modules():
        if (torch.nn.utils.prune.is_pruned(module)):
            param_names = [x for (x, _) in module.named_parameters(recurse= False)]
            for param_name in param_names:
                if (param_name.endswith("_orig")):
                    param = param_name.removesuffix("_orig")
                    torch.nn.utils.prune.remove(module, param)

