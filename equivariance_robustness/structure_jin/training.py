import numpy as np
import torch
from robograph.model.gnn import train
from torch_geometric.data import DataLoader
from tqdm import tqdm


def eval_classifier(model, loader):
    """ Evaluate model with dataloader

    Parameters
    ----------
    model: GC_NET instance
    loader: torch.util.data.DataLoader
        DataLoader with each data in torch.Data
    testing: bool
        Flag for testing. Default: False
    save_path: str
        Load model from saved path. Default: None
    robust: bool
        Flag for robust training. Defualt: False

    Returns
    -------
    accuracy: float
        Accuracy with loader
    """
    model.eval()
    _device = next(model.parameters()).device

    correct = 0
    for data in loader:
        data = data.to(_device)
        output = model(data)
        pred = output.argmax(dim=1)
        correct += pred.eq(data.y).sum().item()

    return correct / len(loader.dataset)


def train_classifier(model, num_epochs, batch_size, train_dataset, val_dataset, test_dataset):
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)


    train_acc_history = []
    val_acc_history = []
    best_state_dict = None

    for epoch in tqdm(range(num_epochs)):
        loss_all = train(model, train_loader)
        train_acc = eval_classifier(model, train_loader)
        val_acc = eval_classifier(model, val_loader)
        if (epoch == 0) or (val_acc >= np.max(val_acc_history)):
            best_state_dict = model.state_dict()
        
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        #tqdm.write("epoch {:03d} ".format(epoch+1) + 
        #        "train_loss {:.4f} ".format(loss_all) +
        #        "train_acc {:.4f} ".format(train_acc) +
        #        "val_acc {:.4f} ".format(val_acc))
    
    model.load_state_dict(best_state_dict)
    test_acc = eval_classifier(model, test_loader)
    tqdm.write("test_acc {:.4f}".format(test_acc))

    return train_acc_history, val_acc_history, test_acc