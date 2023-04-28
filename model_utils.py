import torch
from torch.nn.utils import prune
def prepare_model(model):
    """
    This function is required to be called before several other functions.
    Do not call this function multiple times on the same model.
    Do not prune a model before calling this function on that model.

    This function iterates through all of the parameters of a model and applies a global unstructured identity prune. 
    This does not change the behavior of the model, but makes sure that there are masks for each parameter.
    Since Identity prune is being used, then all the values of all the masks are set to 1 as to not prune the parameters yet.

    
    model: a pytorch model
    returns: a list of strings that are possible parameters to modify, like 'weight' and 'bias'
    """

    # iterate through all of the parameters 
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
    """
    Gets the accuracy of the model when testing the data from the dataloader

    model: The pytorch model
    dataloader: A dataloader that contains all the data for the model to be tested on
    device: the device of the model. That is, a torch.device. Make sure that the whole model is on the same device
    returns: a value from 0..1 being the accuracy percentage
    """

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
    """
    !!! Requires prepare_model to be called on model before use !!!
    Calculates the percentage of the parameters of model that are pruned out

    model: a pytorch model
    param_name: the name of the parameters that are to be checked. These are usually either 'weight' or 'bias'. 
        The return value of prepare_model will list any other parameters to be checked
    returns: a value from 0..1 being the percentage of that parameter type that is pruned
    """

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
    """
    !!! Requires prepare_model to be called on model before use !!!
    This function removes pruning from a model by permanently applying whatever the prune
    This function effectively undoes prepare_model, so prepare_model must be called again before functions that require it

    model: A pytorch model
    """

    # Requires prepare_model
    for module in model.modules():
        if (torch.nn.utils.prune.is_pruned(module)):
            param_names = [x for (x, _) in module.named_parameters(recurse= False)]
            for param_name in param_names:
                if (param_name.endswith("_orig")):
                    param = param_name.removesuffix("_orig")
                    torch.nn.utils.prune.remove(module, param)

