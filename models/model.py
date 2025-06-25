import typing as tp
from abc import ABC, abstractmethod
from flax import nnx


class BaseModel(nnx.Module, ABC):
    """Abstract base class for all models in AetherCV.
    
    This class defines the interface that all models must implement,
    ensuring consistency across different model architectures.
    """
    
    def __init__(self, *, num_classes: int, input_channels: int, rngs: nnx.Rngs):
        """Initialize the base model.
        
        Args:
            num_classes: Number of output classes
            input_channels: Number of input channels
            rngs: Random number generators for parameter initialization
        """
        self.num_classes = num_classes
        self.input_channels = input_channels
        super().__init__()
    
    @abstractmethod
    def __call__(self, x, training: bool = False, return_activations_for_layer: tp.Optional[str] = None):
        """Forward pass through the model.
        
        Args:
            x: Input tensor
            training: Whether the model is in training mode
            return_activations_for_layer: Optional layer name to return activations for
            
        Returns:
            Model output or activations for specified layer
        """
        pass
    
    @abstractmethod
    def get_available_layers(self) -> tp.List[str]:
        """Get list of available layer names for activation extraction.
        
        Returns:
            List of layer names that can be used with return_activations_for_layer
        """
        pass
    
    def validate_layer_name(self, layer_name: str) -> bool:
        """Validate if a layer name is available in this model.
        
        Args:
            layer_name: Name of the layer to validate
            
        Returns:
            True if layer exists, False otherwise
        """
        return layer_name in self.get_available_layers()

def create_model(model_class_name: str, **kwargs) -> BaseModel:
    """Factory function to create models by name.
    
    Args:
        model_class_name: Name of the model class to create
        **kwargs: Arguments to pass to the model constructor
        
    Returns:
        Instantiated model
        
    Raises:
        ValueError: If model_class_name is not recognized
    """
    # Import here to avoid circular imports
    from models.aether import YatCNN, YatResNet
    from models.linear import LinearCNN, LinearResNet
    
    model_classes = {
        'YatCNN': YatCNN,
        'LinearCNN': LinearCNN,
        'YatResNet': YatResNet,
        'LinearResNet': LinearResNet,
    }
    
    if model_class_name not in model_classes:
        available_models = list(model_classes.keys())
        raise ValueError(f"Unknown model class '{model_class_name}'. Available models: {available_models}")
    
    return model_classes[model_class_name](**kwargs)
