import torch

from ai.rl.env.buffer import SmartReplayBuffer


def test_collect_10_s10():
    buffer = SmartReplayBuffer(10, (1,), 1)

    for i in range(10):
        buffer.store((i, i + 1, i + 2, i + 3, int(i % 4 == 0)))

    assert buffer["obs"].shape == (10, 1)
    assert buffer["action"].shape == (10, 1)
    assert buffer["next_obs"].shape == (10, 1)
    assert buffer["reward"].shape == (10,)
    assert buffer["done"].shape == (10,)
    assert buffer.step_ == 10
    assert len(buffer) == 10

    assert buffer.episode_steps_ == [0, 4, 8]
    assert buffer._pointer == 0


def test_collect_15_s10():
    buffer = SmartReplayBuffer(10, (1,), 1)

    for i in range(15):
        buffer.store((i, i + 1, i + 2, i + 3, int(i % 4 == 0)))

    assert buffer["obs"].shape == (10, 1)
    assert buffer["action"].shape == (10, 1)
    assert buffer["next_obs"].shape == (10, 1)
    assert buffer["reward"].shape == (10,)
    assert buffer["done"].shape == (10,)
    assert buffer.step_ == 15

    assert buffer.episode_steps_ == [8, 2]
    assert buffer._pointer == 5

    data = torch.tensor([11, 12, 13]).float().unsqueeze(dim=-1)
    assert torch.isclose(buffer[1:4]["obs"], data).all()

    (traj,) = buffer.trajectories(1)
    traj["obs"].shape == (4, 1)
    assert torch.isclose(
        traj["obs"], torch.tensor([9, 10, 11, 12]).float().unsqueeze(dim=-1)
    ).all()

    assert traj["done"][:-1].sum() == 0
    assert traj["done"][-1] == 1

    trajs = list(buffer.trajectories(4))
    assert len(trajs) == 4
