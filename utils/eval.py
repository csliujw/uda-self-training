import logging

from data.loveda import LoveDALoader
from module.tta import tta, Scale
logger = logging.getLogger(__name__)
from utils.tools import *
from tqdm import tqdm
import ever as er


def evaluate(model, cfg, logger=None):
    model.eval()
    logger.info(cfg.EVAL_DATA_CONFIG)
    eval_dataloader = LoveDALoader(cfg.EVAL_DATA_CONFIG)
    metric_op = er.metric.PixelMetric(len(COLOR_MAP.keys()), logdir=cfg.SNAPSHOT_DIR, logger=logger)
    with torch.no_grad():
        for ret, ret_gt in tqdm(eval_dataloader):
            ret = ret.to(torch.device('cuda'))
            cls = model(ret)
            if isinstance(cls,tuple):
                cls = cls[0]
            cls = cls.argmax(dim=1).cpu().numpy()

            cls_gt = ret_gt['cls'].cpu().numpy().astype(np.int32)
            mask = cls_gt >= 0

            y_true = cls_gt[mask].ravel()
            y_pred = cls[mask].ravel()
            metric_op.forward(y_true, y_pred)
    metric_op.summary_all()
    torch.cuda.empty_cache()
