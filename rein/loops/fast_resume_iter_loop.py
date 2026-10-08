from mmengine.logging import print_log
from mmengine.registry import LOOPS
from mmengine.runner.loops import IterBasedTrainLoop

@LOOPS.register_module()
class FastResumeIterBasedTrainLoop(IterBasedTrainLoop):

    def run(self):
        self.runner.call_hook("before_train")
        self.runner.call_hook("before_train_epoch")

        if self._iter > 0:
            print_log(
                f"Fast resume at iter {self._iter}: skip dataloader replay and "
                "start from a fresh shuffled data stream.",
                logger="current",
            )

        while self._iter < self._max_iters and not self.stop_training:
            self.runner.model.train()
            data_batch = next(self.dataloader_iterator)
            self.run_iter(data_batch)

            self._decide_current_val_interval()
            if (
                self.runner.val_loop is not None
                and self._iter >= self.val_begin
                and (
                    self._iter % self.val_interval == 0
                    or self._iter == self._max_iters
                )
            ):
                self.runner.val_loop.run()

        self.runner.call_hook("after_train_epoch")
        self.runner.call_hook("after_train")
        return self.runner.model
