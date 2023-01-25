from typing import Dict, Optional, Tuple, TypedDict, Union

from ignite.engine import Engine
from ignite.metrics import Loss
import torch

from ..data.build import BatchDataTorch
from ..models.build import BaseModel
from ..utils.metrics import MetricFunction
from ..utils.tensor import prepare_tensor


def prepare_batch(batch: BatchDataTorch, device: Optional[Union[str, torch.device]] = None,
                  non_blocking: bool = False) -> BatchDataTorch:
    """Prepare batch for training: pass to a device with options."""
    batch['x'] = prepare_tensor(batch['x'], device=device, non_blocking=non_blocking)
    batch['y'] = prepare_tensor(batch['y'], device=device, non_blocking=non_blocking)
    batch['m'] = prepare_tensor(batch['m'], device=device, non_blocking=non_blocking)
    return batch


class EngineOutput(TypedDict):
    y_pred: Dict
    y: torch.Tensor
    loss: Optional[torch.Tensor]
    aux: Dict


def y_from_engine(x: EngineOutput) -> Tuple[Dict, torch.Tensor]:
    """Access prediction and ground-truth labels from engine output."""
    return x['y_pred'], x['y']


def loss_from_engine(x: EngineOutput) -> Optional[torch.Tensor]:
    """Access loss output from engine output."""
    return x['loss']


def aux_from_engine(x: EngineOutput) -> Dict:
    """Access auxiliary data from engine output."""
    return x['aux']


def create_trainer(model: BaseModel, optimizer: torch.optim.Optimizer,
                   loss_fn: MetricFunction,
                   metrics: Optional[Dict[str, Loss]] = None,
                   device: Optional[Union[str, torch.device]] = None,
                   non_blocking: bool = False,
                   accumulation_steps: int = 1) -> Engine:
    """Create training engine."""
    if metrics is None:
        metrics = dict()

    if device:
        model.to(device)

    def _update(engine: Engine, batch: BatchDataTorch) -> EngineOutput:
        model.train()

        batch = prepare_batch(batch, device=device, non_blocking=non_blocking)
        # print("batch : ", batch['y'])
        # print("batch : ", batch['y'].shape)
        # batch :  tensor([[ 9.9992e-01,  2.6114e-04,  1.9579e-03, -1.2599e-02,  1.0083e-05,
        #         5.2599e-01,  5.6490e-03,  1.2581e-02],
        #         [ 9.9999e-01,  1.3821e-03,  3.7447e-03, -2.0768e-03, -1.0103e-03,
        #         7.6712e-01, -8.3696e-03,  8.9373e-03],
        #         [ 9.9997e-01, -3.2622e-05,  5.4963e-04,  7.6416e-03, -3.5356e-05,
        #         3.4594e-01,  7.8579e-03,  5.5382e-03],
        #         [ 9.9997e-01, -2.2440e-03, -1.9467e-03, -7.0007e-03,  3.8954e-04,
        #         1.8990e-01,  1.4827e-02, -9.3503e-03],
        #         [ 1.0000e+00,  2.7579e-03, -1.2894e-04,  7.3963e-04, -1.1697e-03,
        #         4.2199e-01, -2.2843e-03,  7.5636e-03]], device='cuda:0')
        # batch :  torch.Size([5, 8])

        if model.has_loss():
            y_pred, loss, _ = model(batch['x'], m=batch['m'], y=batch['y'])
        else:
            y_pred, _, _ = model(batch['x'], m=batch['m'])
            loss = loss_fn(y_pred, batch['y'])

        # print("y_pred : ", y_pred)
        # print("loss : ", loss)

        # check loss
        if loss is None or loss < 0.0:
            raise ValueError("Invalid loss: {}".format(loss.item()))

        loss = loss / accumulation_steps
        loss.backward()

        if engine.state.iteration % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        aux = {'m': batch['m'], 'd': batch['d'], 't': batch['t']}
        output: EngineOutput = {'y_pred': y_pred, 'y': batch['y'], 'loss': loss, 'aux': aux}
        return output

    train_engine = Engine(_update)

    for name, metric in metrics.items():
        metric.attach(train_engine, name)

    return train_engine


def create_evaluator(model: BaseModel,
                     metrics: Optional[Dict[str, Loss]] = None,
                     device: Optional[Union[str, torch.device]] = None, non_blocking: bool = False) -> Engine:
    """Create evaluation engine."""
    if metrics is None:
        metrics = dict()

    if device:
        model.to(device)

    def _inference(_engine: Engine, batch: BatchDataTorch) -> EngineOutput:
        model.eval()
        with torch.no_grad():
            batch = prepare_batch(batch, device=device, non_blocking=non_blocking)
            y_pred, _, _ = model(batch['x'], m=batch['m'])
            aux = {'m': batch['m'], 'd': batch['d'], 't': batch['t']}
            output: EngineOutput = {'y_pred': y_pred, 'y': batch['y'], 'loss': None, 'aux': aux}
            return output

    eval_engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(eval_engine, name)

    return eval_engine
