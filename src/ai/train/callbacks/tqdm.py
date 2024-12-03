from lightning.pytorch.callbacks import TQDMProgressBar


class CustomTQDMProgressBar(TQDMProgressBar):
    def get_metrics(self, trainer, pl_module):
        metrics = super().get_metrics(trainer, pl_module)
        # Format all float metrics to 3 decimal places
        metrics = {
            key: f"{value:.3f}" if isinstance(value, float) else value
            for key, value in metrics.items()
        }
        return metrics
