import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from itertools import product
import argparse
import csv
import tqdm

def pytorch_train_one_epoch(pytorch_network, train_loader, optimizer, loss_function, device):
    """
    Trains the neural network for one epoch on the train DataLoader.
    
    Args:
        pytorch_network (torch.nn.Module): The neural network to train.
        optimizer (torch.optim.Optimizer): The optimizer of the neural network
        loss_function: The loss function.
    
    Returns:
        A tuple (loss, accuracy) corresponding to an average of the losses and
        an average of the accuracy, respectively, on the train DataLoader.
    """
    pytorch_network.train(True)
    with torch.enable_grad():
        loss_sum = 0.
        example_count = 0
        for (x, y) in train_loader:
            # Transfer batch on GPU if needed.
            x = x.to(device)
            y = y.to(device)

            # We need to zero the gradient before every batch because the new
            # gradients would otherwise be summed with the previous gradients.
            optimizer.zero_grad()

            # Compute the predictions of the neural network on the batch.
            y_pred = pytorch_network(x)
            
            loss = loss_function(y_pred, y)
            
            # Do the the backpropagation to compute the gradients of the parameters.
            loss.backward()

            # Update our parameters with the gradient.
            optimizer.step()

            # Since the loss and accuracy are averages for the batch, we multiply 
            # them by the the number of examples so that we can do the right 
            # averages at the end of the epoch.
            loss_sum += float(loss) * len(x)
            example_count += len(x)

    avg_loss = loss_sum / example_count
    return avg_loss

def pytorch_test(pytorch_network, loader, loss_function, device):
    """
    Tests the neural network on a DataLoader.
    
    Args:
        pytorch_network (torch.nn.Module): The neural network to test.
        loader (torch.utils.data.DataLoader): The DataLoader to test on.
        loss_function: The loss function.
    
    Returns:
        A tuple (loss, accuracy) corresponding to an average of the losses and
        an average of the accuracy, respectively, on the DataLoader.
    """
    pytorch_network.eval()
    with torch.no_grad():
        loss_sum = 0.
        example_count = 0
        for (x, y) in loader:
            # Transfer batch on GPU if needed.
            x = x.to(device)
            y = y.to(device)
            
            y_pred = pytorch_network(x)
            loss = loss_function(y_pred, y)

            # Since the loss and accuracy are averages for the batch, we multiply 
            # them by the the number of examples so that we can do the right 
            # averages at the end of the test.
            loss_sum += float(loss) * len(x)
            example_count += len(x)
    
    avg_loss = loss_sum / example_count
    return avg_loss
        
    
def pytorch_train(pytorch_network, 
                  device, 
                  train_loader, 
                  valid_loader, 
                  test_loader, 
                  learning_rate,
                  log_every_n_epochs,
                  num_epochs,
                  logging_dir,
                  job_index):   
    """
    This function transfers the neural network to the right device, 
    trains it for a certain number of epochs, tests at each epoch on
    the validation set and outputs the results on the test set at the
    end of training.
    
    Args:
        pytorch_network (torch.nn.Module): The neural network to train.
    """
    logging_path = logging_dir + f"logs_{job_index}.csv"
    with open(logging_path, "w") as f:
        logger = csv.writer(f)
        logger.writerow(["Epoch", "Train Loss", "Valid Loss"])

    print(f"Logging will be done to path '{logging_path}'")
    print()

    print("Network:")
    print(pytorch_network)
    print()
    
    # Transfer weights on GPU if needed.
    pytorch_network.to(device)
    
    loss_function = nn.MSELoss()
    optimizer = optim.SGD(pytorch_network.parameters(), lr=learning_rate)
    
    for epoch in tqdm.tqdm(list(range(1, num_epochs + 1)), "Processing epochs..."):
        # Training the neural network via backpropagation
        train_loss = pytorch_train_one_epoch(pytorch_network, train_loader, optimizer, loss_function, device)
        
        # Validation at the end of the epoch
        valid_loss = pytorch_test(pytorch_network, valid_loader, loss_function, device)

        if epoch % log_every_n_epochs == 0:
            # NOTE On écrit sur le disque plutôt que juste faire imprimer les valeurs
            with open(logging_path, "a") as f:
                logger = csv.writer(f)
                logger.writerow([epoch, train_loss, valid_loss])
    
    # Test at the end of the training
    test_loss = pytorch_test(pytorch_network, test_loader, loss_function, device)
    print('Test Loss: {}'.format(test_loss))

def get_pytorch_network(model, num_features):
    if model == "A":
        return nn.Sequential(
            nn.Linear(num_features, 1),
            nn.Flatten(0)
        )
    elif model == "B":
        return nn.Sequential(
            nn.Linear(num_features, 1000),
            nn.Linear(1000, 1),
            nn.Flatten(0)
        )
    elif model == "C":
        return nn.Sequential(
            nn.Linear(num_features, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1),
            nn.Flatten(0)
        )
    elif model == "D":
        return nn.Sequential(
            nn.Linear(num_features, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1),
            nn.Flatten(0)
        )
    else:
        raise NotImplementedError(f"No model found for {model}")

def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    # NOTE: Remplacement parce qu'on n'a pas accès à Internet
    X = np.load(config.dataset_path + "X.npy")
    y = np.load(config.dataset_path + "y.npy")

    X_train_valid, X_test, y_train_valid, y_test = train_test_split(X, 
                                                                    y, 
                                                                    train_size=1 - config.test_size, 
                                                                    random_state=config.random_seed)

    scaler = StandardScaler()
    X_train_valid = scaler.fit_transform(X_train_valid)
    X_test = scaler.transform(X_test)

    X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid,
                                                          y_train_valid, 
                                                          train_size=1 - config.valid_size, 
                                                          random_state=config.random_seed)

    print("Training set shapes (X, y):", X_train.shape, y_train.shape)
    print("Validation set shapes (X, y):", X_valid.shape, y_valid.shape)
    print("Testing set shapes (X, y):", X_test.shape, y_test.shape)
    print()

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    valid_dataset = TensorDataset(torch.from_numpy(X_valid), torch.from_numpy(y_valid))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    pytorch_network = get_pytorch_network(config.model, config.num_features)

    pytorch_train(pytorch_network, 
                  device, 
                  train_loader, 
                  valid_loader, 
                  test_loader,
                  config.learning_rate,
                  config.log_every_n_epochs,
                  config.num_epochs,
                  config.logging_path,
                  config.job_index)


def set_config_from_index(args):
    # On veut tester pour 5 random seeds et 4 modèles différents
    seeds = list(range(5))
    models = ["A", "B", "C", "D"]

    # Product va produire un liste [(0, "A"), (0, "B"), ..., (4, "D")] de configs (seed, model)
    configs = list(product(seeds, models))
    config = configs[args.job_index] # On prend celle qui correspond à notre index

    # On ajuste correctement la configuration
    args.random_seed = config[0]
    args.model = config[1]

    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # On gère les emplacements de lecture comme des paramètres pour Calcul Canada
    parser.add_argument("--dataset_path", type=str, default="dataset/")
    parser.add_argument("--logging_path", type=str, default="logging/")

    # Arguments par défaut dans le script original
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--num_features", type=int, default=13)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--num_epochs", type=int, default=10000)
    parser.add_argument("--log_every_n_epochs", type=int, default=100)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--valid_size", type=float, default=0.25)
    parser.add_argument("--model", type=str, default="A")
    
    # On passe un index de job qui va déterminer quelle config on veut lancer précisément
    parser.add_argument("--job_index", type=int, default=-1)

    args = parser.parse_args()

    # Par défaut, l'index est à -1, alors on peut tester comme on veut.
    # Par contre, quand on va sur Calcul Canada, on veut seulement que l'index
    # choisisse la configuration, comme on va toutes les tester.
    if args.job_index != -1:
        args = set_config_from_index(args)

    # On imprime tous les paramètres pour valider
    print("Running with the following config :\n")
    for key, value in args.__dict__.items():
        print(f"{key}: {value}")
    print()

    # On appelle notre script principal avec les arguments sélectionnés
    main(args)