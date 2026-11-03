import os
import numpy as np
for attr, target in [
    ('float', float), ('int', int), ('bool', bool), ('complex', complex),
    ('object', object), ('str', str), ('float_', np.float64),
    ('complex_', np.complex128), ('unicode_', np.str_), ('bool8', np.bool_)
]:
    if not hasattr(np, attr):
        setattr(np, attr, target)
import torch
_orig_torch_load = torch.load
def _safe_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs and 'weights_only' in _orig_torch_load.__code__.co_varnames:
        kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _safe_torch_load
from recbole.config.configurator import Config as RecBole_Config
from recbole.utils import ModelType as RecBoleModelType
from recbole_gnn.utils import get_model, ModelType

class Config(RecBole_Config):
    def __init__(self, model=None, dataset=None, gpu=None, lr=None, weight_decay=None, tem=None, config_file_list=None, config_dict=None):
        """
        Args:
            model (str/AbstractRecommender): the model name or the model class, default is None, if it is None, config
            will search the parameter 'model' from the external input as the model name or model class.
            dataset (str): the dataset name, default is None, if it is None, config will search the parameter 'dataset'
            from the external input as the dataset name.
            config_file_list (list of str): the external config file, it allows multiple config files, default is None.
            config_dict (dict): the external parameter dictionaries, default is None.
        """
        self.gpu = gpu
        super(Config, self).__init__(model, dataset, config_file_list, config_dict)
        self._update_lr(lr=lr, weight_decay=weight_decay)
        if tem:
            self._update_tem(tem)
            
    def _update_tem(self, temperature=0.2):
        self.final_config_dict["temperature"] = float(temperature)
    
    def _update_lr(self, lr=None, weight_decay=None):
        if lr is not None:
            self.final_config_dict["learning_rate"] = float(lr)
        if weight_decay is not None:
            self.final_config_dict["weight_decay"] = float(weight_decay)
        
    def _init_device(self):
        if self.gpu:
            self.final_config_dict["gpu_id"] = self.gpu 
        elif isinstance(self.final_config_dict["gpu_id"], tuple):
            self.final_config_dict["gpu_id"] = ",".join(
                map(str, list(self.final_config_dict["gpu_id"]))
            )
        else:
            self.final_config_dict["gpu_id"] = str(self.final_config_dict["gpu_id"])
        gpu_id = self.final_config_dict["gpu_id"]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ["CUDA_VISIBLE_DEVICES"]))

        if "local_rank" not in self.final_config_dict:
            self.final_config_dict["single_spec"] = True
            self.final_config_dict["local_rank"] = 0
            self.final_config_dict["device"] = (
                torch.device("cpu")
                if len(gpu_id) == 0 or not torch.cuda.is_available()
                else torch.device("cuda")
            )
        else:
            assert len(gpu_id.split(",")) >= self.final_config_dict["nproc"]
            torch.distributed.init_process_group(
                backend="nccl",
                rank=self.final_config_dict["local_rank"]
                + self.final_config_dict["offset"],
                world_size=self.final_config_dict["world_size"],
                init_method="tcp://"
                + self.final_config_dict["ip"]
                + ":"
                + str(self.final_config_dict["port"]),
            )
            self.final_config_dict["device"] = torch.device(
                "cuda", self.final_config_dict["local_rank"]
            )
            self.final_config_dict["single_spec"] = False
            torch.cuda.set_device(self.final_config_dict["local_rank"])
            if self.final_config_dict["local_rank"] != 0:
                self.final_config_dict["state"] = "error"
                self.final_config_dict["show_progress"] = False
                self.final_config_dict["verbose"] = False

    def _get_model_and_dataset(self, model, dataset):

        if model is None:
            try:
                model = self.external_config_dict['model']
            except KeyError:
                raise KeyError(
                    'model need to be specified in at least one of the these ways: '
                    '[model variable, config file, config dict, command line] '
                )
        if not isinstance(model, str):
            final_model_class = model
            final_model = model.__name__
        else:
            final_model = model
            final_model_class = get_model(final_model)

        if dataset is None:
            try:
                final_dataset = self.external_config_dict['dataset']
            except KeyError:
                raise KeyError(
                    'dataset need to be specified in at least one of the these ways: '
                    '[dataset variable, config file, config dict, command line] '
                )
        else:
            final_dataset = dataset

        return final_model, final_model_class, final_dataset
    
    def _load_internal_config_dict(self, model, model_class, dataset):
        super()._load_internal_config_dict(model, model_class, dataset)
        current_path = os.path.dirname(os.path.realpath(__file__))
        model_init_file = os.path.join(current_path, './properties/model/' + model + '.yaml')
        quick_start_config_path = os.path.join(current_path, './properties/quick_start_config/')
        sequential_base_init = os.path.join(quick_start_config_path, 'sequential_base.yaml')
        social_base_init = os.path.join(quick_start_config_path, 'social_base.yaml')

        if os.path.isfile(model_init_file):
            config_dict = self._update_internal_config_dict(model_init_file)

        self.internal_config_dict['MODEL_TYPE'] = model_class.type
        if self.internal_config_dict['MODEL_TYPE'] == RecBoleModelType.SEQUENTIAL:
            self._update_internal_config_dict(sequential_base_init)
        if self.internal_config_dict['MODEL_TYPE'] == ModelType.SOCIAL:
            self._update_internal_config_dict(social_base_init)