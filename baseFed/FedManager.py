import logging
import copy
import math

import numpy as np

from data_preprocessing.loader import Data_Loader
from .globalmodel import GlobalModel
from .server import Server
from .localmodel import LocalModel
from .client import Client
from .message import Message


class FedManager(object):
    def __init__(self, device, args) -> None:
        self.args = args
        self.device = device
        self._setup_dataset()
        self._setup_server()
        self._setup_clients()

    def _setup_dataset(self):
        self.data_loader = Data_Loader(self.args)
        [train_data_num, test_data_num, train_data_global, test_data_global,
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num, other_params] = self.data_loader.load_data()

        self.other_params = other_params
        self.train_global = train_data_global
        self.test_global = test_data_global

        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num

        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict

        self.local_label_data_ds_dict = self.other_params["local_label_data_ds_dict"]
        self.local_unlabel_data_ds_dict = self.other_params["local_unlabel_data_ds_dict"]
        self.train_ori_data_np_dict = self.other_params["train_ori_data_np_dict"]
        self.train_ori_targets_np_dict = self.other_params["train_ori_targets_np_dict"]
        self.train_dl_with_noise_label_dict = self.other_params["train_dl_with_noise_label_dict"]
        self.net_dataidx_map = self.other_params["net_dataidx_map"]
        self.label_noise_whole_dataset = self.other_params["label_noise_whole_dataset"]
        self.real_y = self.other_params["real_y"]
        self.train_targets_with_label_noise_dict = self.other_params["train_targets_with_label_noise_dict"]
        self.class_num = class_num
    
    def _setup_server(self):
        global_model = GlobalModel(self.args)

        self.server = Server(global_model, 'server', 0, None, self.test_global)

    def _setup_clients(self):
        self.client_list = []
        for client_index in range(self.args.client_num):
            localmodel = LocalModel(self.args)
            client = Client('client', client_index, localmodel, 
                            train_data_num=self.train_data_local_num_dict[client_index],
                            train_dl=self.train_data_local_dict[client_index], 
                            test_dl=self.test_data_local_dict[client_index],
                            local_label_data_ds=self.local_label_data_ds_dict[client_index], 
                            local_unlabel_data_ds=self.local_unlabel_data_ds_dict[client_index],
                            train_ori_data_np=self.train_ori_data_np_dict[client_index],
                            train_ori_targets_np=self.train_ori_targets_np_dict[client_index],
                            train_dl_with_noise_label=self.train_dl_with_noise_label_dict[client_index],
                            train_ori_targets_with_label_noise=self.train_targets_with_label_noise_dict[client_index]
                            )
            self.client_list.append(client)
            
    def train(self):
        for round in range(self.args.global_round):

            logging.info("################Communication round : {}".format(round))
            downloaded_model_params = copy.deepcopy(self.server.compressed_global_model_param())

            # ----------------- sample clinet saving in manager------------------#
            client_indexes = self.server.client_sampling(    # 每一轮Sample一些client
                round, self.args.client_num,
                self.args.client_num_per_round)
            distribute_content = {'GLOBAL_PARAM': downloaded_model_params}
            global_message = Message(distribute_content)
            # -----------------train model using algorithm_train and aggregate------------------#
            self.train_locally_per_round(round, client_indexes, global_message)
            # self.train_locally_per_round_FedAvg_lower_bound(round, downloaded_model_params)
            # -----------------aggregation procedure------------------#
            self.server.aggregation()
    
            self.server.test(round, self.test_global, self.device)

    def train_locally_per_round(self, round, selected_clients, global_message):
        uploaded_weights = []
        uploaded_models_params = []
        add_client_index = []
        for num, client_index in enumerate(selected_clients):
            # Update client config before training
            # get the one of the current selected client
            client = self.client_list[client_index]
            # if client.label_flag == False and self.args.model != 'SemiFed':
            #     logging.info('client {} have no label'.format(client.client_index))
            #     continue
            client_message = client.run_train(round, global_message, client_index, self.device)
            # upload_info = client.run_train_FixMatch(round, copy_downloaded_model_params)
            add_client_index.append(client_index)
            uploaded_models_params.append(client_message.content['MODEL_PARAM'])
            uploaded_weights.append(client_message.content['SAMPLE_NUM'])
            # if client.label_flag == True:
            #     uploaded_class_centroids.append(upload_info['CLASS_CENTROID'])
            #     uploaded_class_centroid_weights.append(upload_info['SAMPLE_NUM'])
        logging.info(f'This Round {round}, updated clients is {str(add_client_index)}')
        uploaded_info_for_server = {'CLIENTS_WEIGHTS':uploaded_weights,
                                    'MODELS_PARAMS': uploaded_models_params}

        self.server.receive_message(uploaded_info_for_server)
        logging.info("sampling client_indexes = %s finished the update and upload the info" % str(selected_clients))