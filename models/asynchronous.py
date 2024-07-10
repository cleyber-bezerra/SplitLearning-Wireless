import torch

class AsynchronousSplitLearning:
    def __init__(self, client_models, server_model, num_epoch, num_batch, K, lthred):
        self.state = 'A'
        self.client_models = client_models
        self.server_model = server_model
        self.num_epoch = num_epoch
        self.num_batch = num_batch
        self.K = K
        self.lthred = lthred
        self.total_loss = 0

    def split_forward(self, state, data, target, criterion):
        if state == 'C':
            act, y_star = None, None
        else:
            act = sum(client_model(data) for client_model in self.client_models) / len(self.client_models)
            y_star = target
        outputs = self.server_model(act)
        loss = criterion(outputs, target)
        return loss

    def split_backward(self, state, loss, optimizer):
        loss.backward()
        optimizer.step()

    def update_state(self, total_loss):
        last_update_loss = total_loss / (self.num_batch * self.K)
        delta_loss = last_update_loss - (total_loss / (self.num_batch * self.K))
        if delta_loss <= self.lthred:
            self.state = 'A'
        else:
            self.state = 'B' if self.state == 'A' else 'C'
        return self.state

    def train(self, train_loader, criterion, optimizer, device):
        for epoch in range(1, self.num_epoch + 1):
            total_loss = 0
            for client in range(1, self.K + 1):
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()

                    loss = self.split_forward(self.state, data, target, criterion)
                    total_loss += loss.item()
                    self.split_backward(self.state, loss, optimizer)
            self.state = self.update_state(total_loss)
