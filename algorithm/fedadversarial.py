import torch

from .fedbase import BasicClient, BasicServer


class Server(BasicServer):
    def __init__(self, option, model, clients, test_data=None):
        super().__init__(option, model, clients, test_data)


class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super().__init__(option, name, train_data, valid_data)
        self.gradL = None
        self.alpha = option['alpha']
        self.noise_sd = option['noise_sd']

    def train(self, model):
        model.train()
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        for iter in range(self.epochs):
            for batch_id, batch_data in enumerate(data_loader):
                model.zero_grad()
                inputs = batch_data + torch.rand_like(batch_data, device=self.calculator.get_device()) * self.noise_sd
                loss = self.calculator.get_loss(model, inputs)
                loss.backward()
                optimizer.step()
        return


