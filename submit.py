import argparse
import cv2 as cv
from data.loveda import LoveDALoader
from module.UperNetConvnext import UperNetConvnext
from utils.tools import *
from utils.tools import COLOR_MAP
from data.loveda import LoveDALoader
from utils.tools import *
import simplecv as scv
from utils.tools import COLOR_MAP


palette = np.asarray(list(COLOR_MAP.values())).reshape((-1,)).tolist()
viz_op = scv.viz.VisualizeSegmm('./log/', palette)


def submit_convx(save_dir, pth, default_cfg):
    parser = argparse.ArgumentParser(description='submit.')
    parser.add_argument('--config_path', type=str,  default=default_cfg, help='config path')
    args = parser.parse_args()
    os.makedirs(save_dir,exist_ok=True)
    cfg = import_config(args.config_path)
    model = UperNetConvnext().cuda()
    model.eval()
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model.load_state_dict(torch.load(pth))
    test_dataloader = LoveDALoader(cfg.TEST_DATA_CONFIG)
    with torch.no_grad():
        for ret,ret_gt in tqdm(test_dataloader):
            ret = ret.to(torch.device('cuda'))
            cls = model(ret)
            if isinstance(cls,tuple):
                cls = cls[0]
            cls = cls.argmax(dim=1).cpu().numpy()
            for index in range(len(ret_gt['fname'])):
                cv.imwrite(f'./{save_dir}/{ret_gt["fname"][index]}',cls[index])
                

def evaluate(default_cfg,pth):
    parser = argparse.ArgumentParser(description='Run uda methods.')
    parser.add_argument('--config_path', type=str,default=default_cfg, help='config path')
    args = parser.parse_args()
    cfg = import_config(args.config_path)
    model = UperNetConvnext().cuda()
    model.eval()
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model.load_state_dict(torch.load(pth))
    model.eval()

    logger = get_console_file_logger(name='uda', logdir=cfg.SNAPSHOT_DIR)
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


def show(save_dir, pth, default_cfg):
    parser = argparse.ArgumentParser(description='vis.')
    parser.add_argument('--config_path', type=str,default=default_cfg,
                    help='config path')
    args = parser.parse_args()
    cfg = import_config(args.config_path)
    print(save_dir)
    print(pth)
    model = UperNetConvnext().cuda()
    model.eval()
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model.load_state_dict(torch.load(pth))
    test_dataloader = LoveDALoader(cfg.TEST_DATA_CONFIG)
    with torch.no_grad():
        for ret,ret_gt in tqdm(test_dataloader):
            ret = ret.to(torch.device('cuda'))
            cls = model(ret)
            if isinstance(cls,tuple):
                cls = cls[0]
            cls = cls.argmax(dim=1).cpu().numpy()
            for index in range(len(ret_gt['fname'])):
                cv.imwrite(f'./{save_dir}/{ret_gt["fname"][index]}',cls[index])


if __name__ == '__main__':
    seed_torch(2333)