
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import MSELoss
from open_gns.models import EncodeProcessDecode
from open_gns.dataset import GNSDataset
from torch_geometric.data import DataLoader
from livelossplot import PlotLosses
from tqdm import tqdm


def train(model, train_dataset, val_dataset=None, device=None, num_epochs=10):
    device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    dataloaders = {'train': DataLoader(train_dataset, batch_size=2, shuffle=True)}
    if val_dataset is not None:
        dataloaders['validation'] = DataLoader(val_dataset, batch_size=2, shuffle=False)
    input_size = train_dataset.num_node_features
    optimizer = Adam(model.parameters(), lr=0.001)
    lr_scheduler = ExponentialLR(optimizer, 0.4)
    mse = MSELoss()
    liveloss = PlotLosses()

    for epoch in range(num_epochs):
        logs = {}
        for phase in dataloaders:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            for i, data in enumerate(tqdm(dataloaders[phase])):
                data = data.to(device)
                y_pred = model(data.x, data.edge_index)
                loss = mse(y_pred, data.y)
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item()*data.num_graphs
            # Log epoch loss
            epoch_loss = running_loss/len(dataloaders[phase].dataset)
            label = 'log loss' if phase == 'train' else 'val_log loss'
            logs[label] = epoch_loss
            # Save checkpoint
            if phase == 'train':
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, f'checkpoint_{epoch}_{epoch_loss}.pt')
        lr_scheduler.step()
        liveloss.update(logs)
        liveloss.send()


if __name__ == '__main__':
    train_dataset = GNSDataset('./datasets/box_bath', filename='box_bath_tiny.hdf5', rollouts=(0, 1000))
    val_dataset = GNSDataset('./datasets/box_bath', filename='box_bath_tiny.hdf5', rollouts=(1000, 1150))
    model = EncodeProcessDecode(input_size)
    train(model, train_dataset, val_dataset)

