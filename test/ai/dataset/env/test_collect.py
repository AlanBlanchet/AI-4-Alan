import torch

from ai.dataset.env import ReplayBuffer
from ai.utils.func import TensorInfo


def generate_buffer(
    capacity: int,
    obs_shape: tuple = (),
    out_action: tuple = (),
    done_fn=lambda x: x % 4 == 0,
    loop=None,
):
    loop = loop or capacity

    buffer = ReplayBuffer(
        capacity=capacity,
        obs_info=TensorInfo(shape=obs_shape),
        action_info=TensorInfo(shape=out_action),
    )

    for i in range(loop):
        buffer.store((i, i + 1, i + 2, i + 3, int(done_fn(i))))

    return buffer


def test_collect_10_s10():
    buffer = generate_buffer(10)

    assert buffer["obs"].shape == (10,)
    assert buffer["action"].shape == (10,)
    assert buffer["next_obs"].shape == (10,)
    assert buffer["reward"].shape == (10,)
    assert buffer["done"].shape == (10,)
    assert buffer._step == 10
    assert len(buffer) == 10

    assert buffer._episode_steps == [0, 4, 8]
    assert buffer._ptr == 0


def test_collect_15_s10():
    buffer = generate_buffer(10, loop=15)

    assert buffer["obs"].shape == (10,)
    assert buffer["action"].shape == (10,)
    assert buffer["next_obs"].shape == (10,)
    assert buffer["reward"].shape == (10,)
    assert buffer["done"].shape == (10,)
    assert buffer._step == 15

    assert buffer._episode_steps == [8, 2]
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
    buffer = generate_buffer(10, loop=15)

    assert buffer.sample(3)["obs"].shape == (3, 1)  # (batch, history)


def test_collect_history():
    buffer = generate_buffer(7, loop=6)
    # Get the first 2 elements in history
    obs = buffer.last(3)["obs"][0:2]
    assert obs.allclose(torch.zeros_like(obs))

    # No obs shound be zeros
    buffer = generate_buffer(7, loop=9)
    state = buffer.last(4)
    obs = state["obs"]
    for o in obs:
        assert not o.allclose(torch.zeros_like(o))
    assert state["done"][-1] == 1

    # Just before done should be returned only
    buffer = generate_buffer(7, loop=10)
    obs = buffer.last(4)["obs"]
    assert obs[:-1].allclose(torch.zeros_like(obs[:-1]))
    assert obs[-1] == 9


def test_collect_next_obs():
    buffer = ReplayBuffer(capacity=10)

    buffer.store((1, 2, 4, 10, 0))
    buffer.store((10, 2, 4, 3, 0))

    traj = buffer.last(history=2)
    obs = traj["obs"]
    next_obs = traj["next_obs"]

    assert obs[1].allclose(next_obs[0])


def test_collect_batch_history():
    buffer = ReplayBuffer(capacity=4)

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
    buffer = ReplayBuffer(capacity=4)

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
    buffer = ReplayBuffer(capacity=2)

    buffer.store((1, 2, 4, 10, 0))

    traj = buffer.last(2)

    obs = traj["obs"]

    assert obs[0].allclose(torch.tensor([0]).float())
