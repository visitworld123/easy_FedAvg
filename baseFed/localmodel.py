from model.resnet import ResNet18, ResNet50
import torch
from torch import nn
from utils.average import AverageMeter
from utils.metric import *
from utils.record import *
import copy

class LocalModel(object):
    def __init__(self, args) -> None:
        self.args = args
        self.model_name = self.args.model
        self.model = self._build_model()
        self.optimizer = self._build_optimizer(self.model.parameters())
        self.criterion = self._build_criterion()
        self.local_epoch_num = 0
       

    def get_model_params(self):
        self.model.cpu()
        return copy.deepcopy(self.model.state_dict())
    
    def set_model_params(self, params):
        self.model.load_state_dict(params)

    def _build_model(self):
        if self.model_name == 'resnet18':
            net = ResNet18(self.args, self.args.num_classes, self.args.input_channels)
        elif self.model_name == 'resnet50':
            net = ResNet50(self.args, self.args.num_classes, self.args.input_channels)
        else:
            NotImplementedError
        return net
    
    def _build_optimizer(self, params_to_optimizer):
        if self.args.opti == 'SGD':
            optimizer = torch.optim.SGD(params_to_optimizer,
                                        lr=self.args.lr, 
                                        weight_decay=self.args.wd, 
                                        momentum=self.args.momentum, 
                                        nesterov=self.args.nesterov)
        else:
            NotImplementedError
        return optimizer

    def _build_criterion(self):
        if self.args.task == 'classification':
            criterion = nn.CrossEntropyLoss()
        else:
            NotImplementedError
        return criterion
    
    def local_train_one_epoch(self, round, dataloader, device, **kwargs):
        self.model.to(device)
        self.model.train()


        if self.args.fedprox:
            # previous_model = copy.deepcopy(self.trainer.get_model_params())
            previous_model = kwargs["previous_model"]
        else:
            previous_model = None
        epoch_loss_avg = AverageMeter()
        epoch_acc_avg = AverageMeter()
        total = 0
        correct = 0
        for idx, (x,y) in enumerate(dataloader):
            x, labels = x.to(device), y.to(device)

            self.optimizer.zero_grad()

            real_bs = x.size(0)

            if x.shape[1] == 1:
                assert self.args.dataset in ["mnist", "femnist", "fmnist", "femnist-digit"]
                x = x.repeat(1, 3, 1, 1)
            
            if self.args.model_out_feature:
                output, feat = self.model(x)
            else:
                output = self.model(x)

            loss = self.criterion(output, labels)
            prec1, corr_num, _ = accuracy(output.data, labels)
            correct += corr_num
            if self.args.fedprox:
                fed_prox_reg = 0.0
                previous_model = kwargs["previous_model"]
            
                for name, param in self.model.named_parameters():
                    fed_prox_reg += ((self.args.fedprox_mu / 2) * \
                        torch.norm((param - previous_model[name].data.to(device)))**2)
                loss += fed_prox_reg
        
            total += real_bs
            loss.backward()
            loss_value = loss.item()
            self.optimizer.step()
            epoch_loss_avg.update(loss_value, real_bs)
            epoch_acc_avg.update(prec1, real_bs)


        acc = correct / total * 100
        self.local_epoch_num += 1

        scalar_dict = {'FedAvg_{role}:{index} averager acc'.format(role=kwargs['role'], index=kwargs['index']):epoch_acc_avg.avg,
                       'FedAvg_{role}:{index} loss'.format(role=kwargs['role'], index=kwargs['index']):epoch_loss_avg.avg}
        
        self.record(scalar = scalar_dict, step = self.local_epoch_num)
        
    def record(self,  **kwargs):
        if 'scalar' in kwargs:
            for keys in kwargs['scalar']:
                log_info('scalar', keys ,kwargs['scalar'][keys], kwargs['step'], self.args.record_tool)
        
        if 'image' in kwargs:
            for keys in kwargs['scalar']:
                log_info('image', keys ,kwargs['image'][keys], kwargs['step'], self.args.record_tool)

    
    def test(self, round, dataloader, device, **kwargs):
        
        self.model.eval()


        test_loss_avg = AverageMeter()
        test_acc_avg = AverageMeter()
        total = 0
        correct = 0
        self.model.to(device)
        with torch.no_grad():
            for idx, (x,y) in enumerate(dataloader):
                x, labels = x.to(device), y.to(device)

                real_bs = x.size(0)

                if x.shape[1] == 1:
                    assert self.args.dataset in ["mnist", "femnist", "fmnist", "femnist-digit"]
                    x = x.repeat(1, 3, 1, 1)
                
                if self.args.model_out_feature:
                    output, feat = self.model(x)
                else:
                    output = self.model(x)

                loss = self.criterion(output, labels)
                prec1, corr_num, _ = accuracy(output.data, labels)

                correct += corr_num
                total += real_bs
                loss_value = loss.item()
                test_loss_avg.update(loss_value, real_bs)
                test_acc_avg.update(prec1, real_bs)


        acc = correct / total * 100
        self.local_epoch_num += 1
        scalar_dict = {'FedAvg_{role}:{index} averager acc'.format(role=kwargs['role'], index=kwargs['index']):test_acc_avg.avg,
                       'FedAvg_{role}:{index} real acc'.format(role=kwargs['role'], index=kwargs['index']):acc,
                       'FedAvg_{role}:{index} loss'.format(role=kwargs['role'], index=kwargs['index']):test_loss_avg.avg}
        if self.args.wandb_record:
            self.record(scalar = scalar_dict, step = self.local_epoch_num)

        return test_acc_avg.avg



    
