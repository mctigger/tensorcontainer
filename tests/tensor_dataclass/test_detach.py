import torch

from tensorcontainer.tensor_dataclass import TensorDataClass
from tests.conftest import skipif_no_compile


class TestTensorDataClassDetach:
    def test_detach_returns_new_dataclass_instance(self):
        """Test that detach() creates a new TensorDataClass instance."""
        
        class SimpleData(TensorDataClass):
            observations: torch.Tensor
            actions: torch.Tensor

        # Create TensorDataClass with tensors that don't require gradients
        data = SimpleData(
            observations=torch.randn(4, 10),
            actions=torch.randn(4, 3),
            shape=(4,),
            device=torch.device("cpu"),
        )
        
        detached_data = data.detach()
        
        # Should return a new TensorDataClass instance
        assert isinstance(detached_data, SimpleData)
        assert detached_data is not data
        
        # Data should be preserved
        assert torch.equal(detached_data.observations, data.observations)
        assert torch.equal(detached_data.actions, data.actions)

    def test_detached_tensors_lose_gradients(self):
        """Test that detach() removes gradient tracking from all tensor fields."""
        
        class ModelWeights(TensorDataClass):
            layer1_weights: torch.Tensor
            layer2_weights: torch.Tensor
            bias: torch.Tensor

        # Create TensorDataClass with tensors that require gradients
        weights = ModelWeights(
            layer1_weights=torch.randn(5, 8, requires_grad=True),
            layer2_weights=torch.randn(5, 6, requires_grad=True),
            bias=torch.randn(5, 1, requires_grad=True),
            shape=(5,),
            device=torch.device("cpu"),
        )
        
        detached_weights = weights.detach()
        
        # All detached tensors should not require gradients
        assert not detached_weights.layer1_weights.requires_grad
        assert not detached_weights.layer2_weights.requires_grad  
        assert not detached_weights.bias.requires_grad

    def test_original_dataclass_unchanged_after_detach(self):
        """Test that the original TensorDataClass is not modified by detach()."""
        
        class TrainingState(TensorDataClass):
            parameters: torch.Tensor
            gradients: torch.Tensor

        # Create TensorDataClass with gradient-tracking tensors
        state = TrainingState(
            parameters=torch.randn(3, 7, requires_grad=True),
            gradients=torch.randn(3, 7, requires_grad=True),
            shape=(3,),
            device=torch.device("cpu"),
        )
        
        # Call detach but ignore the result
        state.detach()
        
        # Original tensors should still require gradients
        assert state.parameters.requires_grad
        assert state.gradients.requires_grad

    def test_detached_tensors_share_memory_storage(self):
        """Test that detach() creates tensors sharing the same memory storage."""
        
        class ExperimentData(TensorDataClass):
            features: torch.Tensor
            targets: torch.Tensor

        # Create TensorDataClass with gradient-tracking tensors
        data = ExperimentData(
            features=torch.randn(6, 4, requires_grad=True),
            targets=torch.randn(6, 2, requires_grad=True),
            shape=(6,),
            device=torch.device("cpu"),
        )
        
        detached_data = data.detach()
        
        # Detached tensors should share memory with original tensors
        assert data.features.data_ptr() == detached_data.features.data_ptr()
        assert data.targets.data_ptr() == detached_data.targets.data_ptr()

    def test_detach_preserves_non_tensor_fields(self):
        """Test that detach() preserves non-tensor fields unchanged."""
        
        class ConfigurableModel(TensorDataClass):
            weights: torch.Tensor
            learning_rate: float
            model_name: str
            num_layers: int

        # Create TensorDataClass with mixed field types
        model = ConfigurableModel(
            weights=torch.randn(2, 5, requires_grad=True),
            learning_rate=0.001,
            model_name="transformer", 
            num_layers=6,
            shape=(2,),
            device=torch.device("cpu"),
        )
        
        detached_model = model.detach()
        
        # Non-tensor fields should be preserved exactly
        assert detached_model.learning_rate == 0.001
        assert detached_model.model_name == "transformer"
        assert detached_model.num_layers == 6
        
        # Tensor field should be detached
        assert not detached_model.weights.requires_grad
        assert torch.equal(detached_model.weights, model.weights)

    def test_detach_with_nested_dataclasses(self):
        """Test that detach() works correctly with nested TensorDataClass instances."""
        
        class InnerConfig(TensorDataClass):
            inner_weights: torch.Tensor

        class OuterModel(TensorDataClass):
            outer_weights: torch.Tensor
            config: InnerConfig

        # Create nested TensorDataClass structure
        inner = InnerConfig(
            inner_weights=torch.randn(3, 4, requires_grad=True),
            shape=(3,),
            device=torch.device("cpu"),
        )
        
        model = OuterModel(
            outer_weights=torch.randn(3, 8, requires_grad=True),
            config=inner,
            shape=(3,),
            device=torch.device("cpu"),
        )
        
        detached_model = model.detach()
        
        # Both outer and nested tensors should be detached
        assert not detached_model.outer_weights.requires_grad
        assert not detached_model.config.inner_weights.requires_grad
        
        # Original tensors should still have gradients
        assert model.outer_weights.requires_grad
        assert model.config.inner_weights.requires_grad
        
        # Data should be preserved
        assert torch.equal(detached_model.outer_weights, model.outer_weights)
        assert torch.equal(detached_model.config.inner_weights, model.config.inner_weights)

    @skipif_no_compile
    def test_detach_works_with_torch_compile(self):
        """Test that detach() is compatible with torch.compile."""
        
        class NetworkState(TensorDataClass):
            encoder_weights: torch.Tensor
            decoder_weights: torch.Tensor

        # Create TensorDataClass with gradient-tracking tensors
        network = NetworkState(
            encoder_weights=torch.randn(4, 6, requires_grad=True),
            decoder_weights=torch.randn(4, 3, requires_grad=True),
            shape=(4,),
            device=torch.device("cpu"),
        )

        def detach_operation(dataclass_instance):
            return dataclass_instance.detach()

        # Compile the detach operation
        compiled_detach = torch.compile(detach_operation, fullgraph=True)
        compiled_result = compiled_detach(network)

        # Compiled result should be a proper TensorDataClass
        assert isinstance(compiled_result, NetworkState)
        assert compiled_result is not network
        
        # Compiled result should not have gradient tracking
        assert not compiled_result.encoder_weights.requires_grad
        assert not compiled_result.decoder_weights.requires_grad
        
        # Original should still have gradient tracking
        assert network.encoder_weights.requires_grad
        assert network.decoder_weights.requires_grad
        
        # Should share memory storage
        assert network.encoder_weights.data_ptr() == compiled_result.encoder_weights.data_ptr()
        assert network.decoder_weights.data_ptr() == compiled_result.decoder_weights.data_ptr()
