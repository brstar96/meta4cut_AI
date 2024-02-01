import yaml, os, sys, shutil, datetime, logging
import os.path as osp
sys.path.append('./')
from pytz import timezone


def logger(name, filepath, resume=False):
    lg = logging.getLogger(name)
    lg.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s |[%(lineno)03d]%(filename)-11s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S %Z')
    stream_hd = logging.StreamHandler()
    stream_hd.setFormatter(formatter)
    lg.addHandler(stream_hd)

    file_hd = logging.FileHandler(filepath)
    file_hd.setFormatter(formatter)
    lg.addHandler(file_hd)

    return lg


def parse(args):
    yaml_path = args.yaml_path

    with open(yaml_path, 'r') as fp:
        args = yaml.full_load(fp.read())
    
    name = args['sd_params']['modelname']

    now = datetime.datetime.now(timezone('Asia/Seoul'))
    nowDate = now.strftime('%Y-%m-%d')
    nowTime = now.strftime('%H-%M-%S')
    current_time = now.strftime("%Y_%m_%d_%H_%M_%S")

    # create experiment root
    root = osp.join(args['paths']['experiment_root'], name+'/{}/{}'.format(args['sd_params']['style'], current_time))

    # change runname (train)
    args['runname'] = '{}_style:{}_ver{}_{}'.format(args['sd_params']['modelname'], args['sd_params']['style'], args['version'], current_time)
    
    # general settings
    args['name'] = name
    args['paths']['root'] = root
    args['paths']['visualizations'] = osp.join(root, 'visualization')
    args['paths']['image_results'] = osp.join(root, 'image_results')
    args['paths']['model_save_dir'] = osp.join(root, 'model_save_dir')
    
    for name, path in args['paths'].items():     
        if name == 'state':
            continue
        os.makedirs(path, exist_ok=True)

    lg = logger(args['name'], osp.join(args['paths']['root'], '{}_{}.log'.format(args['name'], current_time)))

    for name, path in args['paths'].items():     
        lg.info('Create directory: {}'.format(path))
    
    return dict_to_nonedict(args), lg, current_time


class NoneDict(dict):
    def __missing__(self, key):
        return None


def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        for k,v in opt.items():
            opt[k] = dict_to_nonedict(v)
        return NoneDict(**opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(x) for x in opt]
    else:
        return opt