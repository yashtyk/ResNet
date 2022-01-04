import logging
from typing import List, Union
import pytorch_lightning as pl
import pl_bolts.optimizers.lr_scheduler
import torch
import torch.nn.functional as F
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay
from pytorch_lightning.core.decorators import auto_move_data
from solarnet.models.backbone import get_backbone
from solarnet.models.classifier import Classifier
from torch import nn, optim
from torchmetrics import Accuracy, F1, MetricCollection, Recall, StatScores
from solarnet.utils.metrics import stats_metrics
from solarnet.models.model_utils import BaseModel
import numpy as np
logger = logging.getLogger(__name__)


class ImageClassification_combine(BaseModel):
    """
    Model for image classification.
    This is a configurable class composed by a backbone (see solarnet.models.backbone.py) and
    a classifier.
    It is also a LightningModule and nn.Module.
    """

    def __init__(
        self,

        n_class: int = 2,

        model1 =None,
        model2 =None,
        model3=None,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.model1= model1
        self.model2= model2




        self.test_metrics = MetricCollection(
            [
                Accuracy(),
                F1(num_classes=self.hparams.n_class, average="macro"),
                Recall(num_classes=self.hparams.n_class, average="macro"),  # balanced acc.
                StatScores(
                    num_classes=self.hparams.n_class if self.hparams.n_class > 2 else 1,
                    reduce="micro",
                    multiclass=self.hparams.n_class > 2,
                ),
            ]
        )



    @property
    def output_size(self) -> int:
        return self.hparams.n_class

    @auto_move_data
    def forward(self, image):
        out1= self.model1(image['magnetogram'])
        out2=self.model2(image[211])
        #out3=self.model3(image[94])
        out1=F.softmax(out1, dim=1)
        out2=F.softmax(out2, dim=1)
        out = out1 + out2
        #out3=F.softmax(out3, dim=1)

        '''
        y_pred1 = torch.argmax(out1, dim=1)
        y_pred2=torch.argmax(out2, dim=1)
        #y_pred3=torch.argmax(out3, dim=1)
        y_pred=y_pred1+y_pred2#+y_pred3
        y_pred = (y_pred>1).int()
        b = np.zeros((y_pred.shape[0], y_pred.max() + 1))
        b[np.arange(y_pred.shape[0]), y_pred] = 1
        res = torch.Tensor(b)
        print(res.shape)
        '''
        return out


    def predict_step(self, batch, batch_idx: int , dataloader_idx: int = None):
        image, _ = batch
        return self(image)



    def test_step(self, batch, batch_idx):
        image, y = batch
        y_pred = self(image)
        y_pred = F.softmax(y_pred, dim=1)

        self.test_metrics(y_pred, y)

    def test_epoch_end(self, outs):
        test_metrics = self.test_metrics.compute()

        tp, fp, tn, fn, _ = test_metrics.pop("StatScores")
        self.log("test_tp", tp)
        self.log("test_fp", fp)
        self.log("test_tn", tn)
        self.log("test_fn", fn)

        for key, value in test_metrics.items():
            self.log(f"test_{key.lower()}", value)






