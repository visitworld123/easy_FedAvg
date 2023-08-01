import logging
import numpy as np

from .message import Message
from .globalmodel import GlobalModel
from  baseFed import strategies

class Server(object):
    def __init__(self, globalmodel:GlobalModel, role, index, train_dl, test_dl) -> None:
        self.globalmodel = globalmodel
        self.role = role
        self.index = index

        self.train_dl = train_dl

        self.test_dl = test_dl

    def receive_message(self, message:Message):
        pass

    def compressed_global_model_param(self):
        # TODO compression algorithm
        return self.globalmodel.get_model_params()
    
    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        # logging.info("Client Sample Probability: %s "%(str(p)))
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            # make sure for each comparison, we are selecting the same clients each round
            num_clients = min(client_num_per_round, client_num_in_total)
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
            # else:
            #     raise NotImplementedError

        logging.info("sampling client_indexes = %s" % str(client_indexes))
        self.selected_clients = client_indexes

        return client_indexes
    
    def test(self, round, testloader, device):
        acc = self.globalmodel.test(round=round, dataloader=testloader, device=device,
                              role=self.role, index=self.index)
        
        logging.info(f"This round: {round} acc is: {acc}")

    def receive_message(self, upload_message):
        # upload_info = {'MODELS_PARAMS':{clinet_index:params},
        #                'CLIENTS_WEIGHTS':{client_index:sample_num}        }

        self.upload_message = upload_message

    def aggregation(self):

        weights = self.upload_message['CLIENTS_WEIGHTS']
        models_params = self.upload_message['MODELS_PARAMS']
        
        logging.info("updata global model")
        model_params = self._federated_averaging_by_params(models_params, weights)
        
        self.globalmodel.set_model_params(model_params)

    def _federated_averaging_by_params(self, models_params, weights):
        model_params = strategies.federated_averaging_by_params(models_params, weights)
        return model_params
    