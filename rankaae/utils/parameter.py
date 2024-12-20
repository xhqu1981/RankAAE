from rankaae.models.model import (
    FCDecoder,
    FCEncoder,
)
import pytorch_optimizer as ex_optim
from torch import optim

AE_CLS_DICT = {
    "FC": {
        "encoder": FCEncoder, 
        "decoder": FCDecoder
    }
}


OPTIM_DICT = {
    "Adam": optim.Adam, 
    "AdamW": optim.AdamW,
    "NAdam": optim.NAdam,
    "SGD": optim.SGD,
    "AdaBound": ex_optim.AdaBound, 
    "RAdam": ex_optim.RAdam,
    "Lamb": ex_optim.Lamb,
    "Soap": ex_optim.SOAP
}


class Parameters():
    
    """
    A parameter object that maps all dictionary keys into its name space.
    The intention is to mimic the functions of a namedtuple.
    """
   
    def __init__(self, parameter_dict):
        
        # "__setattr__" method is changed to immutable for this class.
        super().__setattr__("_parameter_dict", parameter_dict)
        self.update(parameter_dict)
        

    def __setattr__(self, __name, __value):
        """
        The attributes are immutable, they can only be updated using `update` method.
        """
        raise TypeError('Parameters object cannot be modified after instantiation')


    def get(self, key, value):
        """
        Override the get method in the original dictionary parameters.
        """
        return self._parameter_dict.get(key, value)
    

    def update(self, parameter_dict):
        """
        The namespace can only be updated using this method.
        """
        self._parameter_dict.update(parameter_dict)
        self.__dict__.update(self._parameter_dict) # map keys to its name space

    def to_dict(self):
        """
        Return the dictionary form of parameters.
        """
        return self._parameter_dict 


    @classmethod
    def from_yaml(cls, config_file_path):
        """
        Load parameter from a yaml file.
        """
        import yaml

        with open(config_file_path) as f:
            trainer_config = yaml.full_load(f)

        return Parameters(trainer_config)