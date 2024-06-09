from click import command


@command()
def main():
    from ..train.trainer import AITrainer

    trainer = AITrainer("SSD", "detection-datasets/coco")

    trainer.fit()


if __name__ == "__main__":
    main()
