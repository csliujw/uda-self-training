from utils.tools import *
from data.loveda import LoveDALoader
from utils.tools import COLOR_MAP
from tqdm import tqdm
import ever as er
import simplecv as scv
from module.UperNetConvnext import UperNetConvnext
palette = np.asarray(list(COLOR_MAP.values())).reshape((-1,)).tolist()


def visImg(model, cfg, ckpt_path=None, save_dir='./submit_test'):
    viz_op = scv.viz.VisualizeSegmm(save_dir, palette)
    os.makedirs(save_dir, exist_ok=True)
    seed_torch(2333)
    model_state_dict = torch.load(ckpt_path)
    model.load_state_dict(model_state_dict,  strict=True)
    print(ckpt_path)
    # count_model_parameters(model)
    model.eval()
    print(cfg.EVAL_DATA_CONFIG)
    eval_dataloader = LoveDALoader(cfg.EVAL_DATA_CONFIG)

    with torch.no_grad():
        for ret, ret_gt in tqdm(eval_dataloader):
            ret = ret.to(torch.device('cuda'))
            cls = model(ret)[0]
            cls = cls.argmax(dim=1).cpu().numpy()
            for fname, pred in zip(ret_gt['fname'], cls):
                viz_op(pred.astype(np.uint8),fname)
                # imsave(os.path.join(save_dir, fname), pred.astype(np.uint8))
    torch.cuda.empty_cache()

def convNeXt():
    cfg = import_config('st.cbst.2urban')
    model = UperNetConvnext().cuda()
    model.eval()
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
    ckpt_path = '../log/only_source_upernet/2urban/upernet/upernet_conv_2urban_44.6.pth'
    visImg(model, cfg, ckpt_path,save_dir='./save/2urban/convnext/')


if __name__ == '__main__':
    convNeXt()