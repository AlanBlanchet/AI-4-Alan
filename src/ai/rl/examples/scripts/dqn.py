from ai.rl import DQNAgent, Environment, Trainer
from ai.rl.utils.hyperparam import Hyperparam

if __name__ == "__main__":
    # Collect data
    trainer = Trainer(
        DQNAgent(
            Environment("ALE/Pong-v5", memory=150_000),
            epsilon=Hyperparam(0.9, 0.1, steps=100_000),
        ),
        eval_steps=5000,
    )

    paper_iters = 50_000 * 100

    trainer.start(paper_iters)

    trainer.save("pong-v5")
