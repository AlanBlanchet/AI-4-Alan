import numpy as np
import torch
from ai.rl.env.buffer import SmartReplayBuffer


def test_collect_10_s10():
    buffer = SmartReplayBuffer(10, (), 0)

    for i in range(10):
        buffer.store((i, i + 1, i + 2, i + 3, int(i % 4 == 0)))

    assert buffer["obs"].shape == (10,)
    assert buffer["action"].shape == (10,)
    assert buffer["next_obs"].shape == (10,)
    assert buffer["reward"].shape == (10,)
    assert buffer["done"].shape == (10,)
    assert buffer.step_ == 10
    assert len(buffer) == 10

    assert buffer.episode_steps_ == [0, 4, 8]
    assert buffer._ptr == 0


def test_collect_15_s10():
    buffer = SmartReplayBuffer(10, (), 0)

    for i in range(15):
        buffer.store((i, i + 1, i + 2, i + 3, int(i % 4 == 0)))

    assert buffer["obs"].shape == (10,)
    assert buffer["action"].shape == (10,)
    assert buffer["next_obs"].shape == (10,)
    assert buffer["reward"].shape == (10,)
    assert buffer["done"].shape == (10,)
    assert buffer.step_ == 15

    assert buffer.episode_steps_ == [8, 2]
    assert buffer._ptr == 5

    data = torch.tensor([11, 12, 13]).float()
    assert torch.isclose(buffer[1:4]["obs"], data).all()

    (traj,) = buffer.trajectories(1)
    traj["obs"].shape == (4,)
    assert torch.isclose(traj["obs"], torch.tensor([9, 10, 11, 12]).float()).all()

    assert traj["done"][:-1].sum() == 0
    assert traj["done"][-1] == 1

    trajs = list(buffer.trajectories(4))
    assert len(trajs) == 4


def test_collect_15_h1():
    buffer = SmartReplayBuffer(10, (), 0)

    for i in range(15):
        buffer.store((i, i + 1, i + 2, i + 3, int(i % 4 == 0)))

    assert buffer.sample(3)["obs"].shape == (3,)
    assert buffer.sample(3, history=1)["obs"].shape == (3, 1)


def test_collect_15_no_past():
    buffer = SmartReplayBuffer(10, (), 0)

    for i in range(15):
        buffer.store((i, i + 1, i + 2, i + 3, int(i % 3 == 0)))

    np.random.seed(0)
    traj = buffer.sample(1, history=2)
    obs = traj["obs"][0, 0]
    mask = traj["mask"]

    assert obs.allclose(torch.zeros_like(obs))
    assert mask.allclose(torch.tensor([0, 1]).bool())

    traj = buffer.sample(1, history=3)
    obs = traj["obs"][0, :2]
    mask = traj["mask"]

    assert obs.allclose(torch.zeros_like(obs))
    assert mask.allclose(torch.tensor([0, 0, 1]).bool())

    traj = buffer.sample(1, history=4)
    obs = traj["obs"][0, :3]
    mask = traj["mask"]

    assert obs.allclose(torch.zeros_like(obs))
    assert mask.allclose(torch.tensor([0, 0, 0, 1]).bool())


def test_collect_1_mask_start():
    buffer = SmartReplayBuffer(10, (), 0)

    for i in range(1):
        buffer.store((i, i + 1, i + 2, i + 3, int(i % 3 == 0)))

    np.random.seed(6)
    traj = buffer.sample(1, history=2)
    obs = traj["obs"][0, 0]
    mask = traj["mask"]

    assert obs.allclose(torch.zeros_like(obs))
    assert mask.allclose(torch.tensor([0, 1]).bool())


def test_collect_next_obs():
    buffer = SmartReplayBuffer(10, (), 0)

    buffer.store((1, 2, 4, 10, 0))
    buffer.store((10, 2, 4, 3, 0))

    np.random.seed(1)
    traj = buffer.sample(1, history=2)
    obs = traj["obs"]
    next_obs = traj["next_obs"]
    mask = traj["mask"]

    assert obs.allclose(torch.tensor([[1, 10]]).float())
    assert next_obs.allclose(torch.tensor([[10, 3]]).float())
    assert mask.allclose(torch.ones(1, 2).bool())


def test_collect_manually():
    buffer = SmartReplayBuffer(4, (), 0)

    buffer.store((1, 2, 4, 10, 0))
    buffer.store((10, 2, 4, 15, 0))
    buffer.store((15, 2, 4, 20, 1))
    buffer.store((20, 2, 4, 25, 0))
    buffer.store((25, 2, 4, 18, 0))

    exp = buffer[0]
    obs = exp["obs"]
    next_obs = exp["next_obs"]
    done = exp["done"]

    assert obs.allclose(torch.tensor([25]).float())
    assert next_obs.allclose(torch.tensor([18]).float())
    assert done == 0

    exp = buffer[[0, 2]]
    obs = exp["obs"]
    next_obs = exp["next_obs"]
    done = exp["done"]

    assert obs.allclose(torch.tensor([25, 15]).float())
    assert next_obs.allclose(torch.tensor([18, 20]).float())
    assert done.allclose(torch.tensor([0, 1]).float())

    exp = buffer[2:4]
    obs = exp["obs"]
    next_obs = exp["next_obs"]
    done = exp["done"]

    assert obs.shape[0] == 2
    assert obs.allclose(torch.tensor([15, 20]).float())
    assert next_obs.allclose(torch.tensor([20, 25]).float())
    assert done.allclose(torch.tensor([1, 0]).float())


def test_collect_batch_history():
    buffer = SmartReplayBuffer(4, (), 0)

    buffer.store((1, 2, 4, 10, 0))
    buffer.store((10, 2, 4, 15, 0))
    buffer.store((15, 2, 4, 20, 1))
    buffer.store((20, 2, 4, 25, 0))
    buffer.store((25, 2, 4, 18, 0))

    traj = buffer.sample(4, history=4)

    obs = traj["obs"]
    next_obs = traj["next_obs"]
    done = traj["done"]

    assert obs.shape == (4, 4)
    assert next_obs.shape == (4, 4)
    assert done.shape == (4, 4)


def test_get_last():
    buffer = SmartReplayBuffer(4, (), 0)

    buffer.store((1, 2, 4, 10, 0))
    buffer.store((10, 2, 4, 15, 0))
    buffer.store((15, 2, 4, 20, 1))
    buffer.store((20, 2, 4, 25, 0))
    buffer.store((25, 2, 4, 18, 0))

    exp = buffer.last()

    obs = exp["obs"]
    next_obs = exp["next_obs"]
    done = exp["done"]

    assert obs.allclose(torch.tensor([25]).float())
    assert next_obs.allclose(torch.tensor([18]).float())
    assert done == 0

    traj = buffer.last(2)

    obs = traj["obs"]
    next_obs = traj["next_obs"]
    done = traj["done"]

    assert obs.allclose(torch.tensor([20, 25]).float())
    assert next_obs.allclose(torch.tensor([25, 18]).float())
    assert done.allclose(torch.tensor([0, 0]).float())


def test_get_last_1_elem():
    buffer = SmartReplayBuffer(2, (), 0)

    buffer.store((1, 2, 4, 10, 0))

    traj = buffer.last(2)

    obs = traj["obs"]

    assert obs[0].allclose(torch.tensor([0]).float())
