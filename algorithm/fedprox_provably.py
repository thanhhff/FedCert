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
import copy

class Client(BasicClient):
    # TODO: change hard fix options
    # TODO: Tien update, don't have to check
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super().__init__(option, name, train_data, valid_data)
        self.sigma = option['sigma']
        self.N0 = 100
        self.N = 1000
        self.alpha = 0.05
        self.num_classes = option["num_class"]
        self.certify_batch_size = option["certify_batch_size"]
        self.optional_record_path = os.path.join(option['optional_record_path'], option['task'], self.name + '.csv')
        self.mu = option['mu']
        

    def train(self, model):
        # global parameters
        src_model = copy.deepcopy(model)
        src_model.freeze_grad()
        model.train()
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        for iter in range(self.epochs):
            for batch_idx, batch_data in enumerate(data_loader):
                model.zero_grad()
                original_loss = self.calculator.get_loss(model, batch_data)
                # proximal term
                loss_proximal = 0
                for pm, ps in zip(model.parameters(), src_model.parameters()):
                    loss_proximal += torch.sum(torch.pow(pm-ps,2))
                loss = original_loss + 0.5 * self.mu * loss_proximal                #
                loss.backward()
                optimizer.step()
        return

    # TODO: change hard fix options
    def certify(self, model: nn.Module, data_loader: DataLoader) -> pd.DataFrame:
        """
        Return predict, radius
        """
        certify_model = Smooth(model, self.num_classes, self.sigma,
                               self.N0, self.N, self.alpha, self.calculator.device,
                               self.certify_batch_size)
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
                        "correct": correct.item(),
                        "label": output.data.max().item()
                    }
                    certify_results.append(certify_result)
                    idx += 1
        df = pd.DataFrame(certify_results)
        df.to_csv(self.optional_record_path, index=False)
        return df

    def accuracy_at_radii(self, model: nn.Module, data_loader: DataLoader, radii: np.ndarray) -> np.ndarray:
        certify_results = self.certify(model, data_loader)
        accuracy_calculator = ApproximateAccuracy(certify_results)
        return accuracy_calculator.at_radii(radii)

    def certify_train_radii(self, model: nn.Module, radii: np.ndarray):
        data_loader = self.calculator.get_data_loader(
            self.train_data, batch_size=1)
        return self.accuracy_at_radii(model, data_loader, radii)

    def certify_test_radius(self, model: nn.Module, radii: np.ndarray):
        data_loader = self.calculator.get_data_loader(
            self.valid_data, batch_size=1)
        return self.accuracy_at_radii(model, data_loader, radii)


class Server(BasicServer):
    # TODO: change hard fix options
    def __init__(self, option, model: nn.Module, clients: list[Client], test_data=None):
        super().__init__(option, model, clients, test_data)
        self.num_classes = option["num_class"]
        self.sigma = option["sigma"]
        self.N0 = 100
        self.N = 1000
        self.alpha = 0.05
        self.radii = np.arange(0, 1.6, 0.1)
        self.paras_name = ['mu']
        self.certify_batch_size = option['certify_batch_size']
        self.optional_record_path = os.path.join(option['optional_record_path'], option['task'], 'Server.csv')
        
        option_record_folder = os.path.join(option['optional_record_path'], option['task'])
        if not os.path.exists(option_record_folder):
            os.makedirs(option_record_folder)
            
        
        

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

        model_path = os.path.join('fedtask', self.option['task'], 'record', flw.output_filename(self.option, self) + ".pt")
        torch.save(self.model.state_dict(), model_path)

    def log_certify(self):
        server_certify_acc = self.certify().tolist()
        logger.output["server_certify_acc"] = server_certify_acc
        logger.output["client_certify_acc"] = {}

        for idx in range(self.num_clients):
            client_certify_acc = self.clients[idx].certify_test_radius(
                self.model, self.radii)
            logger.output["client_certify_acc"][idx] = client_certify_acc.tolist()

    # TODO: change hard fix options
    def certify(self):
        data_loader = self.calculator.get_data_loader(
            self.test_data, batch_size=self.certify_batch_size)
        certify_model = Smooth(self.model, self.num_classes, self.sigma,
                               self.N0, self.N, self.alpha, device=self.calculator.device,
                               batch_size=self.certify_batch_size)
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
                        "correct": correct.item(),
                        "label": output.data.max().item()
                    }
                    certify_results.append(certify_result)
                    idx += 1
        df = pd.DataFrame(certify_results)

        df.to_csv(self.optional_record_path, index=False)

        # cal accuracy (certify accuracy)
        accuracy_calculator = ApproximateAccuracy(df)
        return accuracy_calculator.at_radii(self.radii)
