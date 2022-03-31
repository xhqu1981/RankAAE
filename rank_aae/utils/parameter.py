from rank_aae.clustering.model import (
    CompactDecoder, 
    CompactEncoder, 
    Encoder, 
    Decoder, 
    QvecDecoder, 
    QvecEncoder, 
    FCDecoder,
    FCEncoder,
)

AE_CLS_DICT = {
    "normal": {
        "encoder": Encoder, 
        "decoder": Decoder
    },
    "compact": {
        "encoder": CompactEncoder, 
        "decoder": CompactDecoder
    },
    "qved": {
        "encoder": QvecEncoder, 
        "decoder": QvecDecoder
    },
    "FC": {
        "encoder": FCEncoder, 
        "decoder": FCDecoder
    }
}


class Parameters():
    """
    A parameter object that maps all dictionary keys into its name space.
    """
    def __init__(self, parameter_dict):
        
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
        return self.__dict__.get(key, value)
    
    def update(self, parameter_dict):
        """
        The namespace can only be updated using this method.
        """
        self.__dict__.update(parameter_dict) # map keys to its name space. 


    @classmethod
    def from_yaml(cls, config_file_path):
        """
        Load parameter from a yaml file.
        """
        import yaml
        with open(config_file_path) as f:
            trainer_config = yaml.full_load(f)
        
        return Parameters(trainer_config)