import argparse
import os.path as osp
import torch.backends.cudnn as cudnn
import torch.optim as optim
import math
from utils.eval import evaluate
from utils.tools import *
from data.loveda import LoveDALoader
from utils.tools import COLOR_MAP
from ever.core.iterator import Iterator
from tqdm import tqdm
from torch.nn.utils import clip_grad
from skimage.io import imsave
from module.UperNetConvnext import UperNetConvnext
import cv2 as cv
import simplecv as scv
from utils.tools import COLOR_MAP

from data.loveda import LoveDALoader
logger = logging.getLogger(__name__)
from utils.tools import *

cudnn.enabled = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True


palette = np.asarray(list(COLOR_MAP.values())).reshape((-1,)).tolist()


def weighted_loss(labels, pred1, pred2):
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
    kl_distance = torch.nn.KLDivLoss(reduction='none')
    loss = criterion(pred1, labels.long())
    sm = torch.nn.Softmax(dim=1)
    log_sm = torch.nn.LogSoftmax(dim=1)

    M = (pred1 + pred2)/2
    variance = torch.sum(kl_distance(log_sm(M),sm(pred1)), dim=1) *0.5 + torch.sum(kl_distance(log_sm(M),sm(pred2)), dim=1)*0.5 
    exp_variance = torch.exp(-variance)
    loss = torch.mean(loss*exp_variance) + torch.mean(variance)
    return loss


def submit_convx(save_dir, pth):
    parser = argparse.ArgumentParser(description='Run uda methods.')
    parser.add_argument('--config_path', type=str,default='st.my.2urban',
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


def show_classifity(save_dir, pth):
    parser = argparse.ArgumentParser(description='Run uda methods.')
    parser.add_argument('--config_path', type=str,default='st.my.2rural',
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

    viz_op = scv.viz.VisualizeSegmm('./log/', palette)

    with torch.no_grad():
        for ret,ret_gt in tqdm(test_dataloader):
            ret = ret.to(torch.device('cuda'))
            cls = model(ret)

            cls1 = cls[0].argmax(dim=1).cpu().numpy()
            cls2 = cls[1].argmax(dim=1).cpu().numpy()
            for index in range(len(ret_gt['fname'])):
                viz_op(cls1[index], "main_"+ret_gt["fname"][index])
                viz_op(cls2[index], "aux_"+ret_gt["fname"][index])
 

def base_train(model,cfg,infomation):
    os.makedirs(cfg.SNAPSHOT_DIR, exist_ok=True)
    logger = get_console_file_logger(name='uda', logdir=cfg.SNAPSHOT_DIR)
    cudnn.enabled = True

    save_pseudo_label_path = osp.join(cfg.SNAPSHOT_DIR, 'pseudo_label')  # in 'save_path'. Save labelIDs, not trainIDs.
    save_stats_path = osp.join(cfg.SNAPSHOT_DIR, 'stats') # in 'save_path'
    
    if not os.path.exists(cfg.SNAPSHOT_DIR):
        os.makedirs(cfg.SNAPSHOT_DIR)
    if not os.path.exists(save_pseudo_label_path):
        os.makedirs(save_pseudo_label_path)
    if not os.path.exists(save_stats_path):
        os.makedirs(save_stats_path)
    
    trainloader = LoveDALoader(cfg.SOURCE_DATA_CONFIG)
    trainloader_iter = Iterator(trainloader)
    evalloader = LoveDALoader(cfg.EVAL_DATA_CONFIG)
    epochs = cfg.NUM_STEPS_STOP / len(trainloader)
    logger.info(infomation)
    logger.info('epochs ~= %.3f' % epochs)
    optimizer = optim.AdamW(model.parameters(),
                          lr=cfg.LEARNING_RATE,  weight_decay=cfg.WEIGHT_DECAY)
    optimizer.zero_grad()
    targetloader = None

    for i_iter in tqdm(range(1,cfg.NUM_STEPS_STOP+1)):
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

        if i_iter % cfg.GENERATE_PSEDO_EVERY == 0 or targetloader is None:
            logger.info('###### Start generate pesudo dataset in round {}! ######'.format(i_iter))
            save_round_eval_path = osp.join(cfg.SNAPSHOT_DIR, str(i_iter))
            save_pseudo_label_color_path = osp.join(save_round_eval_path, 'pseudo_label_color')  # in every 'save_round_eval_path'
            if not os.path.exists(save_round_eval_path):
                os.makedirs(save_round_eval_path)
            if not os.path.exists(save_pseudo_label_color_path):
                os.makedirs(save_pseudo_label_color_path)
            # evaluation & save confidence vectors
            time.sleep(2)
            conf_dict, pred_cls_num, save_prob_path, save_pred_path, image_name_tgt_list = val(model, evalloader, save_round_eval_path, cfg)
            time.sleep(2)
            # class-balanced thresholds
            tgt_portion = min(cfg.TGT_PORTION + cfg.TGT_PORTION_STEP, cfg.MAX_TGT_PORTION)
            cls_thresh = kc_parameters(conf_dict, pred_cls_num, tgt_portion, i_iter, save_stats_path, cfg, logger)
            print('CLS THRESH', cls_thresh)
            time.sleep(2)
            # pseudo-label maps generation
            label_selection(cls_thresh, image_name_tgt_list, i_iter, save_prob_path, save_pred_path, save_pseudo_label_path, save_pseudo_label_color_path, save_round_eval_path, logger)
            time.sleep(2)
            ########### model retraining
            target_config = cfg.TARGET_DATA_CONFIG
            target_config['mask_dir'] = [save_pseudo_label_path]
            logger.info(target_config)
            targetloader = LoveDALoader(target_config)
            targetloader_iter = Iterator(targetloader)
            logger.info('###### Start model retraining dataset in round {}! ######'.format(i_iter))

        model.train()
        lr = adjust_learning_rate(optimizer, i_iter, cfg)

        batch = trainloader_iter.next()
        images_s, labels_s = batch[0]
        main_pred_source, aux_pred_source = model(images_s.cuda())
        batch = targetloader_iter.next()
        images_t, labels_t = batch[0]
        main_pred_target, aux_pred_target = model(images_t.cuda())

        loss_gt = loss_calc(main_pred_source, labels_s['cls'].cuda()) + loss_calc(aux_pred_source, labels_s['cls'].cuda())*0.4
        loss_pseudo = weighted_loss(labels_t['cls'].cuda(),main_pred_target,aux_pred_target) 
        loss = loss_gt + loss_pseudo*0.5

        optimizer.zero_grad()
        loss.backward()
        clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=35, norm_type=2)
        optimizer.step()

        if i_iter % 50 == 0:
            logger.info('exp = {}'.format(cfg.SNAPSHOT_DIR))
            logger.info('Mix iter = %d, loss_gt = %.4f, loss_pseudo = %.4f , lr = %.6f' % (
                i_iter, loss_gt, loss_pseudo, lr))


def train_2rural(pth="./2rural_best.pth"):
    parser = argparse.ArgumentParser(description='Run uda methods.')
    parser.add_argument('--config_path', type=str,default='st.my.2rural',
                   help='config path')
    args = parser.parse_args()
    cfg = import_config(args.config_path)
    cudnn.enabled = True
    model = UperNetConvnext().cuda()
    model.train()
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model.load_state_dict(torch.load(pth))
    base_train(model,cfg,"upernet_2urban")


def train_2urban(pth="./2urban_best.pth"):
    parser = argparse.ArgumentParser(description='Run uda methods.')
    parser.add_argument('--config_path', type=str,default='st.my.2urban',
                   help='config path')
    args = parser.parse_args()
    cfg = import_config(args.config_path)
    cudnn.enabled = True
    model = UperNetConvnext().cuda()
    model.train()
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model.load_state_dict(torch.load(pth))
    base_train(model,cfg,"upernet_2urban")
    

def val(model, targetloader, save_round_eval_path, cfg):
    """Create the model and start the evaluation process."""

    model.eval()
    ## output folder
    save_pred_vis_path = osp.join(save_round_eval_path, 'pred_vis')
    save_prob_path = osp.join(save_round_eval_path, 'prob')
    save_pred_path = osp.join(save_round_eval_path, 'pred')

    viz_op = scv.viz.VisualizeSegmm(save_pred_vis_path, palette)
    # metric_op = er.metric.PixelMetric(len(COLOR_MAP.keys()), logdir=cfg.SNAPSHOT_DIR, logger=logger)

    if not os.path.exists(save_prob_path):
        os.makedirs(save_prob_path)
    if not os.path.exists(save_pred_path):
         os.makedirs(save_pred_path)

    # saving output data
    conf_dict = {k: [] for k in range(cfg.NUM_CLASSES)}
    pred_cls_num = np.zeros(cfg.NUM_CLASSES)
    ## evaluation process
    image_name_tgt_list = []
    with torch.no_grad():
        for batch in tqdm(targetloader):
            images, labels = batch
            output = model(images.cuda())[0].softmax(dim=1)
            output = output[0] if isinstance(output, tuple) else output
            pred_label = output.argmax(dim=1).cpu().numpy()
            output = output.cpu().numpy()
            for fname, pred_i, out_i in zip(labels['fname'], pred_label, output):
                image_name_tgt_list.append(fname.split('.')[0])
                # save prob
                viz_op(pred_i, fname)
                np.save('%s/%s' % (save_prob_path, fname.replace('png', 'npy')), out_i)
                imsave('%s/%s' % (save_pred_path, fname), pred_i.astype(np.uint8), check_contrast=False)
                out_i = out_i.transpose(1,2,0)
                conf_i = np.amax(out_i,axis=2)
                # save class-wise confidence maps
                if cfg.KC_VALUE == 'conf':
                    for idx_cls in range(cfg.NUM_CLASSES):
                        idx_temp = pred_i == idx_cls
                        pred_cls_num[idx_cls] = pred_cls_num[idx_cls] + np.sum(idx_temp)
                        if idx_temp.any():
                            conf_cls_temp = conf_i[idx_temp].astype(np.float32)
                            len_cls_temp = conf_cls_temp.size
                            # downsampling by ds_rate
                            conf_cls = conf_cls_temp[0:len_cls_temp:cfg.DS_RATE]
                            conf_dict[idx_cls].extend(conf_cls)

    return conf_dict, pred_cls_num, save_prob_path, save_pred_path, image_name_tgt_list  # return the dictionary containing all the class-wise confidence vectors


def kc_parameters(conf_dict, pred_cls_num, tgt_portion, round_idx, save_stats_path, cfg, logger):
    logger.info('###### Start kc generation in round {} ! ######'.format(round_idx))
    start_kc = time.time()
    # threshold for each class
    cls_thresh = np.ones(cfg.NUM_CLASSES,dtype = np.float32)
    cls_sel_size = np.zeros(cfg.NUM_CLASSES, dtype=np.float32)
    cls_size = np.zeros(cfg.NUM_CLASSES, dtype=np.float32)
    # if cfg.KC_POLICY == 'cb' and cfg.KC_VALUE == 'conf':
    for idx_cls in np.arange(0, cfg.NUM_CLASSES):
        cls_size[idx_cls] = pred_cls_num[idx_cls]
        if conf_dict[idx_cls] != None:
            conf_dict[idx_cls].sort(reverse=True) # sort in descending order
            len_cls = len(conf_dict[idx_cls])
            cls_sel_size[idx_cls] = int(math.floor(len_cls * tgt_portion))
            len_cls_thresh = int(cls_sel_size[idx_cls])
            if len_cls_thresh != 0:
                cls_thresh[idx_cls] = conf_dict[idx_cls][len_cls_thresh-1]
            conf_dict[idx_cls] = None

    # threshold for mine_id with priority
    num_mine_id = len(np.nonzero(cls_size / np.sum(cls_size) < cfg.MINE_PORT)[0])
    # chose the smallest mine_id
    id_all = np.argsort(cls_size / np.sum(cls_size))
    rare_id = id_all[:cfg.RARE_CLS_NUM]
    mine_id = id_all[:num_mine_id] # sort mine_id in ascending order w.r.t predication portions
    # save mine ids
    np.save(save_stats_path + '/rare_id_round' + str(round_idx) + '.npy', rare_id)
    np.save(save_stats_path + '/mine_id_round' + str(round_idx) + '.npy', mine_id)
    logger.info('Mining ids : {}! {} rarest ids: {}!'.format(mine_id, cfg.RARE_CLS_NUM, rare_id))
    # save thresholds
    np.save(save_stats_path + '/cls_thresh_round' + str(round_idx) + '.npy', cls_thresh)
    np.save(save_stats_path + '/cls_sel_size_round' + str(round_idx) + '.npy', cls_sel_size)
    logger.info('###### Finish kc generation in round {}! Time cost: {:.2f} seconds. ######'.format(round_idx,time.time() - start_kc))
    return cls_thresh


def label_selection(cls_thresh, image_name_tgt_list, round_idx, save_prob_path, save_pred_path, save_pseudo_label_path, save_pseudo_label_color_path, save_round_eval_path, logger,gen_soft=True):
    print("label_selection_save_prob_path",save_prob_path,gen_soft)
    logger.info('###### Start pseudo-label generation in round {} ! ######'.format(round_idx))
    start_pl = time.time()
    viz_op = scv.viz.VisualizeSegmm(save_pseudo_label_color_path, palette)
    for sample_name in image_name_tgt_list:
        probmap_path = osp.join(save_prob_path, '{}.npy'.format(sample_name))
        pred_prob = np.load(probmap_path)
        weighted_prob = pred_prob / cls_thresh[:,None, None]
        weighted_pred_trainIDs = np.asarray(np.argmax(weighted_prob, axis=0), dtype=np.uint8)
        # weighted_conf = np.amax(weighted_prob, axis=0)
        pred_label_trainIDs = weighted_pred_trainIDs.copy()
        pred_label_labelIDs = pred_label_trainIDs + 1
        # 不做标签筛选。
        # pred_label_labelIDs[weighted_conf] = 0 # '0' in LoveDA Dataset ignore
        # pseudo-labels with labelID
        viz_op(pred_label_trainIDs, '%s_color.png' % sample_name)

        # save pseudo-label map with label IDs
        imsave(os.path.join(save_pseudo_label_path, '%s.png' % sample_name), pred_label_labelIDs, check_contrast=False)
        
    # remove probability maps
    if cfg.RM_PROB:
        shutil.rmtree(save_prob_path)

    logger.info('###### Finish pseudo-label generation in round {}! Time cost: {:.2f} seconds. ######'.format(round_idx,time.time() - start_pl))


if __name__ == '__main__':
    seed_torch(2333)
    train_2rural()
    train_2urban()
