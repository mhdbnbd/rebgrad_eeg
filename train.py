import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model.net import Net
from model.rev import GradientReversal
from data.preprocess import load_preprocess

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(batch_size, epochs, subject_id, low_cut_hz):
    """
      trains model
      Parameters:
      -----------------
      batch_size: batch size
      epochs: number of epochs
      subject_id: subject to load (1..9)
      low_cut_hz: low cut frequency for filtering

      Returns:
      -----------------
      trained model prints validation accuracies
    """
    # load model and its components
    model = Net().to(device)
    feature_extractor = model.feature_extractor
    clf = model.classifier

    # create label discriminator
    discriminator = nn.Sequential(
        GradientReversal(),
        nn.Linear(2800, 10),
        nn.ReLU(),
        nn.Linear(10, 4),
        nn.ReLU(),
        nn.Linear(4, 1)
    ).to(device)

    half_batch = batch_size // 2
    # load dataset and create dataloaders
    source_dataset, target_dataset = load_preprocess(subject_id, low_cut_hz)
    source_loader = DataLoader(source_dataset, batch_size=half_batch,
                               shuffle=True, num_workers=1, pin_memory=True)
    target_loader = DataLoader(target_dataset, batch_size=half_batch,
                               shuffle=True, num_workers=1, pin_memory=True)
    # set optimizer
    optim = torch.optim.Adam(list(discriminator.parameters()) + list(model.parameters()))

    for epoch in range(1, epochs + 1):
        batches = zip(source_loader, target_loader)
        n_batches = min(len(source_loader), len(target_loader))
        total_domain_loss = total_label_accuracy = 0
        # Iterate over data
        for (source_x, source_labels), (target_x, _) in batches:
            x = torch.cat([source_x, target_x])
            x = x.to(device)
            x = x.unsqueeze(-1)
            # label source, target domains
            domain_y = torch.cat([torch.ones(source_x.shape[0]),
                                  torch.zeros(target_x.shape[0])])
            domain_y = domain_y.to(device)
            label_y = source_labels.to(device)
            features = feature_extractor(x).view(x.shape[0], -1)
            domain_preds = discriminator(features).squeeze()
            label_preds = clf(features[:source_x.shape[0]])
            # compute losses
            domain_loss = F.binary_cross_entropy_with_logits(domain_preds, domain_y)
            label_loss = F.cross_entropy(label_preds, label_y)
            loss = domain_loss + label_loss
            # zero the parameter gradients
            optim.zero_grad()
            # backward and optimize
            loss.backward()
            optim.step()
            # statistics
            total_domain_loss += domain_loss.item()
            total_label_accuracy += (label_preds.max(1)[1] == label_y).float().mean().item()

        mean_loss = total_domain_loss / n_batches
        mean_accuracy = total_label_accuracy / n_batches
        print(f'EPOCH {epoch:03d}: loss={mean_loss:.4f}, '
                   f'accuracy={mean_accuracy:.4f}')