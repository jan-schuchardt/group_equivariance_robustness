import torch
from gpytorch.distributions import MultivariateNormal

def force_loss(
    pred_energy, 
    pred_forces, 
    target_energy, 
    target_forces, 
    rho_force,
    energy_loss_type="mae", 
    force_loss_type="mae",
):
    if energy_loss_type == "elbo":
        pass
    elif energy_loss_type == "mae":
        energy_mae = torch_mae(targets=target_energy, pred=pred_energy)
    else:
        raise NotImplementedError
    if force_loss_type == "mae":
        force_metric = torch_mae(targets=target_forces, pred=pred_forces)
    elif force_loss_type == "rmse":
        force_metric = get_rmse(targets=target_forces, pred=pred_forces)
    else:
        raise NotImplementedError(f"{force_loss_type} is not implemented")
    loss = energy_mae * (1 - rho_force) + rho_force * force_metric
    return loss

def comb_force_loss(pred_energy, pred_forces, target_energy, target_forces, energy_fn, force_fn, rho_force):
    loss = (1 - rho_force) * energy_fn(pred_energy, target_energy) + rho_force * force_fn(target_forces, pred_forces)
    return loss

def neg_var_elbo(pred, target, elbo):
    return - elbo(pred, target)
    
def get_rmse(targets, pred):
    """
    Mean L2 Error
    """
    return torch.mean(torch.norm((pred - targets), p=2, dim=1))

def torch_mae(pred, targets):
    """
    Mean Absolute Error
    """
    if isinstance(pred, MultivariateNormal):
        pred = pred.mean
    return torch.nn.functional.l1_loss(pred, targets, reduction="mean")
    
def get_model_saving_name(uncertainty_model_name, uncertainty_model_params, encoder_name, encoder_params, target):
    if uncertainty_model_name == 'gp':
        model_saving_name = get_dklp_saving_name(uncertainty_model_name, 'var_elbo', 'gauss_likelihood', uncertainty_model_params)
    return encoder_name + '_' + model_saving_name #get_encoder_saving_name(encoder_name, encoder_params, target)


def get_encoder_saving_name(model_name, model_params, target):
    if model_name == "dimenet":
        model_saving_name = get_dimenet_saving_name(**model_params, target=target)
    elif model_name == "dimenet-mod":
        model_saving_name = get_dimenet_saving_name(**model_params, target=target, comment='MOD')
    else:
        raise ValueError(f"Unknown model name: '{model_name}'")
    return model_saving_name


def get_dimenet_saving_name(hidden_channels, num_bilinear, num_spherical, num_radial, num_blocks, 
                            num_before_skip, num_after_skip, num_output_layers, cutoff, 
                            envelope_exponent, target, comment='', **kwargs):
    model_saving_name = (
            "e" + str(hidden_channels)
            + "_bi" + str(num_bilinear)
            + "_sbf" + str(num_spherical)
            + "_rbf" + str(num_radial)
            + "_b" + str(num_blocks)
            + "_nbs" + str(num_before_skip)
            + "_nas" + str(num_after_skip)
            + "_no" + str(num_output_layers)
            + "_cut" + str(cutoff)
            + "_env" + str(envelope_exponent)
            + "_" + str(comment)
            + "_" + '-'.join(target)
    )
    return model_saving_name


def get_dklp_saving_name(encoder, loss_fn, likelihood, gp_params):
    return (
        "encoder_" + str(encoder)
        + "_gp_params_" + get_name_gp_params(**gp_params)
        + "_loss_function_" + str(loss_fn)
        + "_likelihood" + str(likelihood)
    )

    
def get_name_gp_params(num_outputs, initial_lengthscale, kernel):
    return (
        "num_outputs_" + str(num_outputs)
        + "_initial_lengthscale_" + str(initial_lengthscale)
        + "_kernel_" + str(kernel)
    )
    
def read_json(path):
    """ """
    if not path.endswith(".json"):
        raise UserWarning(f"Path {path} is not a json-path.")

    with open(path, "r") as f:
        content = json.load(f)
    return content


def update_json(path, data):
    """ """
    if not path.endswith(".json"):
        raise UserWarning(f"Path {path} is not a json-path.")

    content = read_json(path)
    content.update(data)
    write_json(path, content)


def write_json(path, data):
    """ """
    if not path.endswith(".json"):
        raise UserWarning(f"Path {path} is not a json-path.")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def read_value_json(path, key):
    """ """
    content = read_json(path)

    if key in content.keys():
        return content[key]
    else:
        return None

def _standardize(kernel):
    """
    Makes sure that Var(W) = 1 and E[W] = 0
    """
    eps = 1e-6

    if len(kernel.shape) == 3:
        axis = [0, 1]  # last dimension is output dimension
    else:
        axis = 1

    var, mean = torch.var_mean(kernel, dim=axis, unbiased=True, keepdim=True)
    kernel = (kernel - mean) / (var + eps) ** 0.5
    return kernel


def he_orthogonal_init(tensor):
    """
    Generate a weight matrix with variance according to He initialization.
    Based on a random (semi-)orthogonal matrix neural networks
    are expected to learn better when features are decorrelated
    (stated by eg. "Reducing overfitting in deep networks by decorrelating representations",
    "Dropout: a simple way to prevent neural networks from overfitting",
    "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks")
    """
    tensor = torch.nn.init.orthogonal_(tensor)

    if len(tensor.shape) == 3:
        fan_in = tensor.shape[:-1].numel()
    else:
        fan_in = tensor.shape[1]

    with torch.no_grad():
        tensor.data = _standardize(tensor.data)
        tensor.data *= (1 / fan_in) ** 0.5

    return tensor

def string_to_dict(data):
    datalist = data[1:-1].split(',')
    my_dict = {}
    for elem in datalist:
        key, data = elem.split(':')
        try: 
            value = int(data)
        except:
            try:
                value = float(data)
            except:
                value = data.strip()[1:-1]
        my_dict[key.strip()[1:-1]] = value
    return my_dict

def list_to_dict(elements):
    my_dict = {}
    for elem in elements:
        try:
            key, data = elem.split("=")
        except:
            continue
        try:
            value = int(data)
            my_dict[key] = value
        except:
            try:
                value = float(data)
                my_dict[key] = value
            except:
                if data[0] == "'":
                    my_dict[key] = str(data[1:-1])
                elif data == 'False' or data == 'True':
                    my_dict[key] = data == 'True'
                elif data[0] == '{':
                    my_dict[key] = string_to_dict(data)
                elif data == 'None':
                    my_dict[key] = None
                else:
                    print(key, data, "are not matching")
    return my_dict