"""Test neural network string representations"""

from ai.nn.arch.resnet.resnet import ResNet18
from ai.nn.compat.module import ModuleList
from ai.nn.modules.conv import ConvBlock


def test_resnet18_representation():
    """Test that ResNet18 shows proper hierarchical representation"""
    resnet = ResNet18()
    repr_str = str(resnet)

    # Should be hierarchical (multi-line)
    assert "\n" in repr_str, "ResNet18 should use hierarchical representation"

    # Should start with class name and parameters
    assert repr_str.startswith("ResNet18("), "Should start with class name"

    # Should show simple properties with equals (only non-default values)
    assert "out_dim=512" in repr_str, "Should show out_dim parameter"
    # first_layer_channels=64 should be hidden as it's the default value

    # Should show modules hierarchically with equals syntax
    assert "input_proj=" in repr_str, "Should show input_proj module"
    assert "layers=" in repr_str, "Should show layers module"

    # Should show nested structure
    assert "  0=" in repr_str, "Should show indexed modules"
    lines = repr_str.split("\n")
    assert len(lines) > 5, "Should be multi-line hierarchical representation"


def test_modulelist_representation():
    """Test that ModuleList shows its contents properly"""
    # Create a simple ModuleList
    modules = ModuleList(
        [
            ConvBlock(3, 64, kernel_size=3),
            ConvBlock(64, 128, kernel_size=3),
        ]
    )

    repr_str = str(modules)

    # Should show contents, not be empty
    assert "ModuleList(" in repr_str, "Should start with ModuleList"
    assert "0=" in repr_str, "Should show first module"
    assert "1=" in repr_str, "Should show second module"
    assert "ConvBlock" in repr_str, "Should show module types"
    # Should be multi-line hierarchical
    assert "\n" in repr_str, "Should be hierarchical representation"

    # Test empty ModuleList
    empty_modules = ModuleList()
    empty_repr = str(empty_modules)
    assert empty_repr == "ModuleList()", "Empty ModuleList should show as ModuleList()"


def test_convblock_representation():
    """Test ConvBlock representation"""
    conv = ConvBlock(3, 64, kernel_size=7, stride=2, padding=3)
    repr_str = str(conv)

    # Should show parameters
    assert "in_channels=3" in repr_str, "Should show in_channels"
    assert "out_channels=64" in repr_str, "Should show out_channels"
    assert "kernel_size=(7, 7)" in repr_str, "Should show kernel_size as tuple"
    assert "stride=2" in repr_str, "Should show stride"
    assert "padding=3" in repr_str, "Should show padding"


def test_resnet18_layers_content():
    """Test that ResNet18 layers contain expected structure"""
    resnet = ResNet18()

    # Should have 4 layers
    assert len(resnet.layers) == 4, "ResNet18 should have 4 layers"

    # Each layer should be a ModuleList
    for i, layer in enumerate(resnet.layers):
        assert isinstance(layer, ModuleList), f"Layer {i} should be ModuleList"
        assert len(layer) == 2, f"Layer {i} should have 2 blocks"

    # First layer should have 64 channels output
    first_layer = resnet.layers[0]
    repr_str = str(first_layer)
    assert "out_channels=[64, 64]" in repr_str, (
        "First layer should have 64 output channels"
    )

    # Last layer should have 512 channels output
    last_layer = resnet.layers[-1]
    repr_str = str(last_layer)
    assert "out_channels=[512, 512]" in repr_str, (
        "Last layer should have 512 output channels"
    )


def test_hierarchical_vs_flat_representation():
    """Test that modules use hierarchical while simple objects use flat representation"""
    resnet = ResNet18()

    # ResNet (module with submodules) should be hierarchical
    resnet_repr = str(resnet)
    assert "\n" in resnet_repr, "ResNet should use hierarchical representation"

    # ConvBlock now uses flat representation (overridden __str__)
    conv = ConvBlock(3, 64)
    conv_repr = str(conv)
    assert "\n" not in conv_repr, "ConvBlock should use flat representation"
    assert conv_repr == "ConvBlock(in_channels=3, out_channels=64)", (
        "ConvBlock should show only non-default values in flat format"
    )

    # Simple MyConf object should be flat
    from myconf import MyConf

    class SimpleConfig(MyConf):
        value: int = 42
        name: str = "test"

    # Test with default values (should show empty)
    simple_default = SimpleConfig()
    default_repr = str(simple_default)
    assert "\n" not in default_repr, "Simple MyConf should use flat representation"
    assert default_repr == "SimpleConfig()", "Default values should be hidden"

    # Test with non-default values
    simple_custom = SimpleConfig(value=100, name="custom")
    custom_repr = str(simple_custom)
    assert "\n" not in custom_repr, "Simple MyConf should use flat representation"
    assert "value=100" in custom_repr, "Should show non-default value"
    assert "name=custom" in custom_repr, "Should show non-default name"
