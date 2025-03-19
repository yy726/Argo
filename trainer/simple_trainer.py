import torch
import torch.nn as nn
import torch.optim as optim


class SimpleTrainer:
    """
        A simple trainer, with minimal training loop logic
    """

    def __init__(self, model, data_loader):
        self.model = model
        self.data_loader = data_loader

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # loss function, for now we use the default BCE
        self.criterion = nn.BCELoss()
        # optimizer, for now we use the default AdamW
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001)

    def train(self, num_epochs: int = 10):
        print("Simple trainer start to train")

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0

            for i, (inputs, labels) in enumerate(self.data_loader):
                # move data to the target device
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                labels = labels.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward pass
                outputs = self.model(inputs).squeeze(1)  # B x 1 shape, remove last dimension
                loss = self.criterion(outputs, labels)

                # backward pass
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if i % 100 == 99:
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(self.data_loader)}], Loss: {running_loss / 100:.4f}')
                    running_loss = 0.0
        
        print("Simple trainer finished training")

        