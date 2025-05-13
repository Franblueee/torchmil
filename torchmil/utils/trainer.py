import torch
from tqdm import tqdm
from copy import deepcopy

import torchmetrics
from torchmil.models import MILModelWrapper, MILModel
from torchmil.utils.annealing_scheduler import AnnealingScheduler


class Trainer:
    """
    Generic trainer class for training MIL models.
    """

    def __init__(
        self,
        model: MILModel,
        optimizer: torch.optim.Optimizer,
        metrics_dict: dict[str : torchmetrics.Metric] = {
            "accuracy": torchmetrics.Accuracy(task="binary"),
        },
        obj_metric: str = "accuracy",
        obj_metric_mode: str = "max",
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        annealing_scheduler_dict: dict[str:AnnealingScheduler] = None,
        device: str = "cuda",
        logger=None,
        early_stop_patience: int = None,
        disable_pbar: bool = False,
        verbose: bool = True,
    ) -> None:
        """
        Arguments:
            model: MIL model to be trained. Must be an instance of [MILModel](../models/mil_model.md).
            optimizer: Optimizer for training the model.
            metrics_dict: Dictionary of metrics to be computed during training. Metrics should be instances of [torchmetrics.Metric](https://torchmetrics.readthedocs.io/en/v0.8.2/references/metric.html).
            obj_metric: Objective metric to be used for early stopping and to track the best model. Must be one of the keys in `metrics_dict`.
            obj_metric_mode: Mode for the objective metric. Must be one of 'max' or 'min'. If 'max', the best model is the one with the highest value of the objective metric. If 'min', the best model is the one with the lowest value of the objective metric.
            lr_scheduler: Learning rate scheduler.
            annealing_scheduler_dict: Dictionary of annealing schedulers for loss coefficients. Keys should be the loss names and values should be instances of [AnnealingScheduler](./annealing_scheduler.md).
            device: Device to be used for training.
            logger: Logger to log metrics. Must have a `log` method. It can be, for example, a [Wandb Run](https://docs.wandb.ai/ref/python/run/).
            early_stop_patience: Patience for early stopping. If None, early stopping is disabled.
            disable_pbar: Disable progress bar.
        """
        self.model = MILModelWrapper(model)
        self.optimizer = optimizer
        self.metrics_dict = metrics_dict
        self.obj_metric_name = obj_metric
        self.obj_metric_mode = obj_metric_mode
        self.lr_scheduler = lr_scheduler
        self.annealing_scheduler_dict = annealing_scheduler_dict
        self.device = device
        self.logger = logger
        self.early_stop_patience = early_stop_patience
        self.disable_pbar = disable_pbar
        self.verbose = verbose
        self.loss_history = {}

        if self.early_stop_patience is None:
            self.early_stop_patience = float("inf")

        if self.obj_metric_mode not in ["max", "min"]:
            raise ValueError(
                f"obj_metric_mode must be one of ['max', 'min'], but got {self.obj_metric_mode}"
            )

        self.best_model_state_dict = None
        self.best_obj_metric = None
        self.model = self.model.to(self.device)

    def _log(self, metrics: dict[str:float]) -> None:
        """
        Log metrics using the logger.

        Arguments:
            metrics: Dictionary of metrics to be logged.
        """
        if self.logger is not None:
            self.logger.log(metrics)
        self._save_metrics_in_history(metrics)

    def _print(self, message: str) -> None:
        """
        Print message if verbose is True.

        Arguments:
            message: Message to be printed.
        """
        if self.verbose:
            print(message)

    def _save_metrics_in_history(self, metrics: dict[str:float]) -> None:
        """
        Save metrics in history.

        Arguments:
            metrics: Dictionary of metrics to be saved.
        """
        for k, v in metrics.items():
            if k not in self.loss_history.keys():
                self.loss_history[k] = []
            self.loss_history[k].append(v)

    def train(
        self,
        max_epochs: int,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader = None,
        test_dataloader: torch.utils.data.DataLoader = None,
    ) -> None:
        """
        Train the model.

        Arguments:
            max_epochs: Maximum number of epochs to train.
            train_dataloader: Train dataloader.
            val_dataloader: Validation dataloader. If None, the train dataloader is used.
            test_dataloader: Test dataloader. If None, test metrics are not computed.
        """

        if val_dataloader is None:
            val_dataloader = train_dataloader

        if self.best_model_state_dict is None:
            self.best_model_state_dict = self.get_model_state_dict()
            if self.obj_metric_mode == "max":
                self.best_obj_metric = float("-inf")
            else:
                self.best_obj_metric = float("inf")
        early_stop_count = 0
        for epoch in range(1, max_epochs + 1):

            # Train loop
            train_metrics = self._shared_loop(
                train_dataloader,
                epoch=epoch,
                mode="train",
            )
            self._log(train_metrics)
            torch.cuda.empty_cache()  # clear cache

            # Validation loop
            val_metrics = self._shared_loop(val_dataloader, epoch=epoch, mode="val")

            self._log(val_metrics)
            torch.cuda.empty_cache()  # clear cache

            # Test loop
            if test_dataloader is not None:
                test_metrics = self._shared_loop(
                    test_dataloader,
                    epoch=epoch,
                    mode="test",
                )
                self._log(test_metrics)
                torch.cuda.empty_cache()  # clear cache

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self._print(
                f'Best {self.obj_metric_name}: {self.best_obj_metric}, Current {self.obj_metric_name}: {val_metrics[f"val/{self.obj_metric_name}"]}'
            )

            if self.obj_metric_mode == "max":
                is_better = (
                    val_metrics[f"val/{self.obj_metric_name}"] > self.best_obj_metric
                )
            else:
                is_better = (
                    val_metrics[f"val/{self.obj_metric_name}"] < self.best_obj_metric
                )

            if not is_better:
                early_stop_count += 1
                self._print(f"Early stopping count: {early_stop_count}")
            else:
                self.best_obj_metric = val_metrics[f"val/{self.obj_metric_name}"]
                self.best_model_state_dict = self.get_model_state_dict()
                early_stop_count = 0

            if early_stop_count >= self.early_stop_patience:
                self._print(f"Reached early stopping condition")
                break

    def _shared_loop(
        self,
        dataloader: torch.utils.data.DataLoader,
        epoch: int = 0,
        mode: str = "train",
    ) -> dict[str:float]:
        """
        Shared training/validation/test loop.

        Arguments:
            dataloader: Dataloader.
            epoch: Epoch number.
            mode: Mode of the loop. Must be one of 'train', 'val', 'test'.
        """

        if mode == "train":
            name = "Train"
        elif mode == "val":
            name = "Validation"
        elif mode == "test":
            name = "Test"

        self.model.train()
        pbar = tqdm(
            enumerate(dataloader), total=len(dataloader), disable=self.disable_pbar
        )
        pbar.set_description(f"[Epoch {epoch}] {name} ")

        loop_metrics_dict = self.metrics_dict
        for k in loop_metrics_dict.keys():
            loop_metrics_dict[k].reset()
        loop_loss_dict = {"loss": torchmetrics.MeanMetric()}

        for batch_idx, batch in pbar:

            batch = batch.to(self.device)

            Y = batch["Y"]  # (batch_size, 1)

            self.optimizer.zero_grad()

            Y_pred, loss_dict = self.model.compute_loss(batch)

            loss = 0.0
            for loss_name, loss_value in loss_dict.items():
                coef = 1.0
                if self.annealing_scheduler_dict is not None:
                    if loss_name in self.annealing_scheduler_dict.keys():
                        coef = self.annealing_scheduler_dict[loss_name]()
                loss += coef * loss_value
                if loss_name not in loop_loss_dict.keys():
                    loop_loss_dict[loss_name] = torchmetrics.MeanMetric()
                loop_loss_dict[loss_name].update(loss_value.item())
            loop_loss_dict["loss"].update(loss.item())

            if mode == "train":

                loss.backward()
                self.optimizer.step()

                if self.annealing_scheduler_dict is not None:
                    for annealing_scheduler in self.annealing_scheduler_dict.values():
                        annealing_scheduler.step()

            for k in loop_metrics_dict.keys():
                loop_metrics_dict[k].update(Y_pred, Y)

            if batch_idx < (len(dataloader) - 1):
                pbar.set_postfix(
                    {
                        f"{mode}/{loss_name}": loop_loss_dict[loss_name]
                        .compute()
                        .item()
                        for loss_name in loss_dict
                    }
                )
            else:
                metrics = {
                    f"{mode}/{k}": v.compute().item() for k, v in loop_loss_dict.items()
                }
                metrics = {
                    **metrics,
                    **{
                        f"{mode}/{k}": v.compute().item()
                        for k, v in loop_metrics_dict.items()
                    },
                }
                pbar.set_postfix(metrics)

            del batch, Y, Y_pred, loss
        pbar.close()
        return metrics

    def get_model_state_dict(self) -> dict:
        """
        Get (a deepcopy of) the state dictionary of the model.

        Returns:
            State dictionary of the model.
        """
        state_dict = deepcopy(self.model.model.state_dict())
        return state_dict

    def get_best_model_state_dict(self) -> dict:
        """
        Get the state dictionary of the best model (the model with the best objective metric).

        Returns:
            State dictionary of the best model.

        """
        return self.best_model_state_dict

    def get_best_model(self) -> MILModel:
        """
        Get the best model (the model with the best objective metric).

        Returns:
            Best model.
        """
        model = self.model.model
        model.load_state_dict(self.best_model_state_dict)
        return model
