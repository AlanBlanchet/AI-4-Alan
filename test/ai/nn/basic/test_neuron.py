from ai.nn.basic.neuron.neuron import LNeuron, Neuron


def test_lneuron():
    model = LNeuron(in_features=5)

    assert model.in_features == 5


def test_neuron():
    model = Neuron(in_features=5)

    assert model.in_features == 5
