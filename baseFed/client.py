import copy

from .localmodel import LocalModel
from .message import Message

class Client(object):
    def __init__(self, role, index, localmodel: LocalModel, train_data_num, train_dl, test_dl,
                local_label_data_ds, local_unlabel_data_ds, train_ori_data_np,
                train_ori_targets_np, train_dl_with_noise_label, train_ori_targets_with_label_noise) -> None:
        self.role = role
        self.index = index
        self.localmodel = localmodel
        self.train_data_num = train_data_num

        self.train_dl = train_dl
        self.test_dl = test_dl
        self.train_ori_data_np = train_ori_data_np
        self.train_ori_targets_np = train_ori_targets_np

        self.local_label_data_ds = local_label_data_ds
        self.local_unlabel_data_ds = local_unlabel_data_ds
        self.train_dl_with_noise_label = train_dl_with_noise_label
        self.train_ori_targets_with_label_noise = train_ori_targets_with_label_noise

    def set_model_params(self, decoded_model_params):
        self.localmodel.set_model_params(decoded_model_params)

    def prepare_message(self):
        parameter = self.localmodel.get_model_params()
        communication_info = {'MODEL_PARAM':parameter, 'SAMPLE_NUM': self.train_data_num}
        communication_message = Message(communication_info)
        return communication_message
    

    def decode_info(self, message: Message):
        # TODO decode algorithm on global distributed model
        if 'GLOBAL_PARAM' in message.content.keys():
            model_params = copy.deepcopy(message.content['GLOBAL_PARAM'])
        return model_params
    
    def run_train(self, round, global_message, client_idx, device):
        decode_global_model = self.decode_info(global_message)

        self.set_model_params(decode_global_model)

        self.localmodel.local_train_one_epoch(round, self.train_dl, device, role='client', index=client_idx)
        

        return self.prepare_message()