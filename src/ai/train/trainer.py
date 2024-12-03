from typing import Any

from lightning.pytorch.loops.evaluation_loop import _EvaluationLoop
from lightning.pytorch.trainer.states import RunningStage, TrainerFn
from lightning.pytorch.trainer.trainer import Trainer


class CustomEvaluationLoop(_EvaluationLoop):
    ...
    # @_no_grad_context
    # def run(self):
    #     self.setup_data()
    #     if self.skip:
    #         return []
    #     self.reset()
    #     self.on_run_start()
    #     data_fetcher = self._data_fetcher
    #     assert data_fetcher is not None
    #     previous_dataloader_idx = 0
    #     print("BLABLa")
    #     exit(0)
    #     while True:
    #         try:
    #             if isinstance(data_fetcher, _DataLoaderIterDataFetcher):
    #                 dataloader_iter = next(data_fetcher)
    #                 # hook's batch_idx and dataloader_idx arguments correctness cannot be guaranteed in this setting
    #                 batch = data_fetcher._batch
    #                 batch_idx = data_fetcher._batch_idx
    #                 dataloader_idx = data_fetcher._dataloader_idx
    #             else:
    #                 dataloader_iter = None
    #                 batch, batch_idx, dataloader_idx = next(data_fetcher)
    #             if previous_dataloader_idx != dataloader_idx:
    #                 # the dataloader has changed, notify the logger connector
    #                 self._store_dataloader_outputs()
    #             previous_dataloader_idx = dataloader_idx
    #             self.batch_progress.is_last_batch = data_fetcher.done
    #             # run step hooks
    #             self._evaluation_step(batch, batch_idx, dataloader_idx, dataloader_iter)
    #         except StopIteration:
    #             print("WRAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAaa")
    #             # this needs to wrap the `*_step` call too (not just `next`) for `dataloader_iter` support
    #             print("MMMMMMMMMMMM")
    #             exit(0)
    #         finally:
    #             self._restarting = False

    # self._store_dataloader_outputs()
    # return self.on_run_end()


class AITrainer(Trainer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.validate_loop = CustomEvaluationLoop(
            self,
            TrainerFn.VALIDATING,
            RunningStage.VALIDATING,
            inference_mode=False,
        )
