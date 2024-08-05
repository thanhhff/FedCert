import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from .fedbase import BasicClient, BasicServer
from .fedprob_utils.smooth import Smooth
from .fedprob_utils.accuracy import ApproximateAccuracy
from main import logger 
import utils.fflow as flw
import pickle as pk 

# SIGMA = 0.5



class Client(BasicClient):
    # TODO: change hard fix options
    # TODO: Tien update, don't have to check 
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super().__init__(option, name, train_data, valid_data)
        self.sigma = option['sigma']    
        self.N0 = 100
        self.N = 1000
        self.alpha = 0.05
        self.num_classes = 10

    def train(self, model: nn.Module):
        """
        Training process for smoothed classifier
        Client training with noisy data
        """
        model.train()
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)

        # Traing phase for base classifer
        for iter in range(self.epochs):
            for batch_id, batch_data in enumerate(data_loader):
                model.zero_grad()
                inputs, outputs = batch_data
                inputs = inputs.to(self.calculator.device) + torch.rand_like(inputs, device=self.calculator.device) * self.sigma

                noisy_batch = [inputs, outputs]

                loss = self.calculator.get_loss(model, noisy_batch)
                loss.backward()
                optimizer.step()

    # TODO: change hard fix options
    def certify(self, model: nn.Module, data_loader: DataLoader) -> pd.DataFrame:
        """
        Return predict, radius
        """
        certify_model = Smooth(model, self.num_classes, self.sigma , self.N0, self.N, self.alpha, self.calculator.device)
        certify_results = []
        idx = 0
        certify_sample = 1

        for batch_id, batch_data in enumerate(data_loader):
            inputs, outputs = batch_data
            batch_size = inputs.shape[0]
            
            for i in range(batch_size):
                if idx % certify_sample == 0:
                    input, output = inputs[i], outputs[i]
                    pred, radius = certify_model.certify(input)
                    correct = pred == output.data.max()
                    certify_result = {
                        "radius": radius,
                        "correct": correct
                    }
                    certify_results.append(certify_result)
                    idx += 1 
        return pd.DataFrame(certify_results)
    
    def accuracy_at_radii(self, model: nn.Module, data_loader: DataLoader, radii: np.ndarray) -> np.ndarray:
        certify_results = self.certify(model, data_loader)
        accuracy_calculator = ApproximateAccuracy(certify_results)
        return accuracy_calculator.at_radii(radii)

    def certify_train_radii(self, model: nn.Module, radii: np.ndarray):
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
        return self.accuracy_at_radii(model, data_loader, radii)

    def certify_test_radius(self, model: nn.Module, radii: np.ndarray):
        data_loader = self.calculator.get_data_loader(self.valid_data, batch_size=self.batch_size)
        return self.accuracy_at_radii(model, data_loader, radii)


class Server(BasicServer):
    # TODO: change hard fix options
    def __init__(self, option, model: nn.Module, clients: list[Client], test_data=None):
        super().__init__(option, model, clients, test_data)
        self.num_classes = 10
        self.sigma = option["sigma"]
        self.N0 = 100
        self.N = 1000
        self.alpha = 0.05
        self.radii = np.arange(0, 1.6, 0.1)
        self.batch_size = 64
        self.num_synth_samples = 64
        self.gen_round = 20
        self.synthetic_data = None
        self.silly_epoch = 5
        self.generate_round = list(range(280, 300))

    def run(self):
        """
        Start the federated learning symtem where the global model is trained iteratively.
        """
        logger.time_start('Total Time Cost')
        for round in range(self.num_rounds+1):
            print("--------------Round {}--------------".format(round))
            logger.time_start('Time Cost')

            # federated train
            self.iterate(round)
            # decay learning rate
            self.global_lr_scheduler(round)

            if (self.synthetic_data != None):
                self.model.train()
                silly_optim = torch.optim.SGD(self.model.parameters(), lr=0.01)
                silly_criterion = nn.CrossEntropyLoss()

                for epoch in range(self.silly_epoch):
                    silly_optim.zero_grad()
                    inputs, outputs = self.synthetic_data
                    inputs = inputs.to(self.calculator.device)
                    outputs = outputs.to(self.calculator.device)
                    preds = self.model(inputs)
                    loss = silly_criterion(preds, outputs)
                    loss.backward()
                    silly_optim.step()
                                    
            if (round in self.generate_round):
                self.gen_synthetic_data()



            logger.time_end('Time Cost')
            if logger.check_if_log(round, self.eval_interval):
                logger.log(self)
            
            

        print("=================End==================")
        self.log_certify()
        logger.time_end('Total Time Cost')
        # save results as .json file
        logger.save(os.path.join('fedtask', self.option['task'], 'record', flw.output_filename(self.option, self)))
        
        f = open( os.path.join('fedtask', self.option['task'], 'record', flw.output_filename(self.option, self)) + '.pkl', "wb")
        pk.dump(self.model, f)

    def gen_synthetic_data(self):
        device = self.calculator.device
        synthetic_sample = torch.randn([self.num_synth_samples, 3, 32, 32], requires_grad=True)
        synthetic_label = torch.empty(self.num_synth_samples, dtype=torch.long).random_(self.num_classes)
        synthetic_label = synthetic_label.to(device)

        synth_optim = torch.optim.SGD([synthetic_sample], lr=0.1)
        synth_criterion = torch.nn.CrossEntropyLoss()
        synth_epochs = 1000

        # Optimize steps
        for epoch in range(synth_epochs):
            synth_optim.zero_grad()
            inputs = synthetic_sample.to(device)
            outputs = self.model(inputs)
            loss = synth_criterion(outputs, synthetic_label)
            loss.backward()
            synth_optim.step()

        self.synthetic_data = (synthetic_sample, synthetic_label)

        
    def log_certify(self):
        server_certify_acc = self.certify().tolist()
        logger.output["server_certify_acc"] = server_certify_acc
        logger.output["client_certify_acc"] = {}

        for idx in range(self.num_clients):
            client_certify_acc = self.clients[idx].certify_test_radius(self.model, self.radii)
            logger.output["client_certify_acc"][idx] = client_certify_acc.tolist()

    # TODO: change hard fix options
    def certify(self):
        data_loader = self.calculator.get_data_loader(self.test_data, batch_size=self.batch_size)
        certify_model = Smooth(self.model, self.num_classes, self.sigma, self.N0, self.N, self.alpha, device=self.calculator.device)
        certify_results = []
        idx = 0
        certify_sample = 1
        

        for batch_id, batch_data in enumerate(data_loader):
            inputs, outputs = batch_data
            batch_size = inputs.shape[0]
            
            for i in range(batch_size):
                if idx % certify_sample == 0:
                    input, output = inputs[i], outputs[i]
                    pred, radius = certify_model.certify(input)
                    correct = pred == output.data.max()
                    certify_result = {
                        "radius": radius,
                        "correct": correct
                    }
                    certify_results.append(certify_result)
                    idx += 1 
        df = pd.DataFrame(certify_results)

        # cal accuracy (certify accuracy)
        accuracy_calculator = ApproximateAccuracy(df)
        return accuracy_calculator.at_radii(self.radii)