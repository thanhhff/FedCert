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
        self.entropy = self.get_entropy()

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
    
    def get_entropy(self) -> float:
        outputs = [d[1] for d in self.train_data]
        labels = np.array(outputs)
        hist, bins = np.histogram(labels, 10, [0, 10])
        hist = hist + 1e-8
        entropy = -np.sum(hist / np.sum(hist) * np.log2(hist / np.sum(hist)))
        return entropy
        


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

        self.client_entropy = [client.entropy for client in clients]
        self.client_entropy = [entropy / sum(self.client_entropy) for entropy in self.client_entropy]
        
    

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
            # self.certify()

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
        
    def log_certify(self):
        server_certify_acc = self.certify().tolist()
        logger.output["server_certify_acc"] = server_certify_acc
        logger.output["client_certify_acc"] = {}

        for idx in range(self.num_clients):
            client_certify_acc = self.clients[idx].certify_test_radius(self.model, self.radii)
            logger.output["client_certify_acc"][idx] = client_certify_acc.tolist()

    def iterate(self, t):
        """
        The standard iteration of each federated round that contains three
        necessary procedure in FL: client selection, communication and model aggregation.
        :param
            t: the number of current round
        """
        # sample clients: MD sampling as default but with replacement=False
        self.selected_clients = self.sample()
        # training
        models, train_losses = self.communicate(self.selected_clients)
        # check whether all the clients have dropped out, because the dropped clients will be deleted from self.selected_clients
        if not self.selected_clients: return
        # aggregate: pk = 1/K as default where K=len(selected_clients)
        self.model = self.aggregate(models, p = [1.0 * self.client_entropy[cid]/self.data_vol for cid in self.selected_clients])
        return
        

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