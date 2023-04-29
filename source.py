import argparse
import os.path as osp
import torch.optim as optim
from data.loveda import LoveDALoader
from module.UperNetConvnext import UperNetConvnext
from utils.tools import *
from utils.tools import COLOR_MAP
import torch.backends.cudnn as cudnn
import cv2 as cv
import torch.backends.cudnn as cudnn
from ever.core.iterator import Iterator
from data.loveda import LoveDALoader
from utils.eval import evaluate
from utils.tools import *

palette = np.asarray(list(COLOR_MAP.values())).reshape((-1,)).tolist()


def base_train(model, cfg, name, single=False):
    """_summary_

    Args:
        model (_type_): network
        cfg (_type_): configuration
        name (_type_): _description_
        single (bool, optional): False mean don't use auxiliary branch
    """
    cudnn.enabled = True
    trainloader = LoveDALoader(cfg.SOURCE_DATA_CONFIG)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE,  weight_decay=cfg.WEIGHT_DECAY)
    cfg.SNAPSHOT_DIR = cfg.SNAPSHOT_DIR + "/" + str(name)
    os.makedirs(cfg.SNAPSHOT_DIR, exist_ok=True)
    logger = get_console_file_logger(name=name, logdir=cfg.SNAPSHOT_DIR)

    count_model_parameters(model, logger)
    epochs = cfg.NUM_STEPS_STOP / len(trainloader)
    logger.info('exp = %s' % cfg.SNAPSHOT_DIR)
    logger.info(model)
    logger.info('epochs ~= %.3f' % epochs)
    
    trainloader_iter = Iterator(trainloader)
    optimizer.zero_grad()
    for i_iter in tqdm(range(cfg.NUM_STEPS_STOP)):
        optimizer.zero_grad()
        lr = adjust_learning_rate(optimizer, i_iter, cfg)
        # Train with Source
        batch = trainloader_iter.next()
        images_s, labels_s = batch[0]
        primary,aux = model(images_s.cuda())
        # Segmentation Loss
        primary_seg = loss_calc(primary, labels_s['cls'].cuda())
        if not single:
            aux_seg = loss_calc(aux, labels_s['cls'].cuda())
        else:
            aux_seg = 0
        loss = primary_seg + aux_seg*0.4
        loss.backward()
        optimizer.step()
            
        if i_iter % 50 == 0:
            logger.info('exp = {}'.format(cfg.SNAPSHOT_DIR))
            logger.info('iter = %d, primary_seg = %.6f, aux_seg = %.6f, lr = %.6f' % (i_iter, primary_seg, aux_seg, lr))
        if i_iter >= cfg.NUM_STEPS_STOP - 1:
            print('save model ...')
            ckpt_path = osp.join(cfg.SNAPSHOT_DIR, cfg.TARGET_SET + str(cfg.NUM_STEPS_STOP) + '.pth')
            torch.save(model.state_dict(), ckpt_path)
            evaluate(model, cfg, logger)
            break
        if i_iter % cfg.EVAL_EVERY == 0 and i_iter != 0:
            ckpt_path = osp.join(cfg.SNAPSHOT_DIR, cfg.TARGET_SET + str(i_iter) + '.pth')
            torch.save(model.state_dict(), ckpt_path)
            evaluate(model, cfg, logger)
            model.train()


def train_2urban(single=False):
    parser = argparse.ArgumentParser(description='Run source methods.')
    parser.add_argument('--config_path', type=str,default='base.2urban',
                   help='config path')
    args = parser.parse_args()
    cfg = import_config(args.config_path)
    cudnn.enabled = True
    model = UperNetConvnext().cuda()
    model.train()
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
    base_train(model, cfg, "upernet_2urban", single)


def train_2rural(single=False):
    parser = argparse.ArgumentParser(description='Run source methods.')
    parser.add_argument('--config_path', type=str,default='base.2rural',
                   help='config path')
    args = parser.parse_args()
    cfg = import_config(args.config_path)
    cudnn.enabled = True
    model = UperNetConvnext().cuda()
    model.train()
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
    base_train(model, cfg, "upernet_2rural", single)


def submit_convx(save_dir, pth, default_cfg):
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('--config_path', type=str,default=default_cfg,
                    help='config path')
    args = parser.parse_args()
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


if __name__ == '__main__':
    seed_torch(2333)
    train_2urban(single=False)
    train_2rural(single=False)
    # submit_convx(save_dir='submit_rural_upernet',pth='../log/only_source_upernet/2rural/upernet/0.43395-test.pth',default_cfg='base.2rural')