import argparse

import torch, os, random, numpy
from torch.utils.data import DataLoader, DistributedSampler

from src.models import UnnamedModel, load_states_from_pretrained_detr
from src.losses import CollectiveLoss
from src.datasets import *
from src.utils import misc as utils
from engine import Trainer


def get_args_parser():
    parser = argparse.ArgumentParser('Training & evaluation configs', add_help=False)

    # global
    parser.add_argument('--device', default='cpu',
                        help='device to use for training / testing')
    #FIXME: evaluation flag: main.py script should be generalized for evaluating as well
    
    # training configs
    ## loss coef: Matcher and Collective Loss share thes params
    parser.add_argument('--cls_coef', default=2, type=float)
    parser.add_argument('--bbox_coef', default=5, type=float)
    parser.add_argument('--giou_coef', default=2, type=float)
    parser.add_argument('--obj_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0., type=float, 
                        help="coef in focal loss, `0` means do not use.")

    ## loss
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int) # per process
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--lr_drop', default=10000, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    
    parser.add_argument('--random_drop', type=float, default=0.1)
    parser.add_argument('--fp_ratio', type=float, default=0.2)
    parser.add_argument('--fp_max_score', type=float, default=0.6, 
                        help="max sampled false positive track score.")
    
    # eval
    parser.add_argument('--eval_freq', default=0, type=int, help="evaluation frequency, 0 means do not evaluate.")
    parser.add_argument('--ckpt_path', default="", type=str, help="path to a checkpoint, '' means training from scratch.")
    parser.add_argument('--save_freq', default=0, type=int, help="checkpoint save frequency, 0 means do not save.")
    parser.add_argument('--save_dir', default="", type=str, 
                        help="path to a checkpoint directory, if not given, create one automatically: `./checkpoints`")
    parser.add_argument('--log_dir', default="", type=str, 
                        help="path to tensorboard log, if not given, does not write log / summary.")

    # model configs
    parser.add_argument('--init_state', nargs="?", const=True, default=False,
                        help="load partial weights from pretrained DETR model (transformer attention, box embed head etc.). \
                            One can pass an url or a path to DETR state dictionary, or does not pass any following value, \
                            in which case weights will be downloaded automatically.")
    ## backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    
    ## transformer
    parser.add_argument('--num_frames', default=4, type=int)
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--update_track_pos', action='store_true', 
                        help="whether update position embeddings of inherited track queries.")
    parser.add_argument('--enc_use_temporal_attn', action='store_true', 
                        help="whether enable temporal attention in encoder layers.")
    parser.add_argument('--no_aux_output', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # NOTE: Deformable attn is not used in this project
    # parser.add_argument('--dec_n_points', default=4, type=int)
    # parser.add_argument('--enc_n_points', default=4, type=int)

    # dataset configs
    parser.add_argument('--dataset_file', type=str, default='mot17', choices=["tao", "mot17", "test"])
    parser.add_argument('--root', type=str, help="frame root")
    parser.add_argument("--anno_root", type=str, help="annotation root")
    parser.add_argument('--val_ratio', default=0.25, type=float, help="validation set ratio")
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--max_sample_interval', default=12, type=int, help="maximum video key frame sampling interval")
    parser.add_argument('--min_sample_interval', default=8, type=int, help="minimum video key frame sampling interval")
    parser.add_argument('--sample_mode', type=str, default="random", choices=["fixed", "random"], help="sampling interval strategy")
    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--num_workers', default=2, type=int) # dataloader workers
    parser.add_argument('--max_size', default=1333, type=int)
    parser.add_argument('--val_width', default=800, type=int)

    parser.add_argument('--num_iters', type=int, default=1, nargs='*')
    parser.add_argument('--resample_epochs', type=int, nargs='*', 
                        help="specify which training epoch the number of iterations will be changed.")
    
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser


def main(args: argparse.Namespace):
    # global
    num_cls_map = {"mot17": 1, "tao": 833, "test": 20}
    assert args.dataset_file in num_cls_map, "Current supported data: `TAO`, `MOT17`, got {}".format(args.dataset_file)

    device = torch.device(args.device)

    num_classes = num_cls_map[args.dataset_file]
    freeze_backbone = args.lr_backbone <= 0.
    use_focal = args.focal_alpha > 0.

    utils.init_distributed_mode(args)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)

    # instantiating model
    # TODO: distributed model
    model = UnnamedModel(args.num_frames, num_classes,
                         d_model=args.hidden_dim, num_heads=args.nheads, 
                         num_encoder_layers=args.enc_layers, num_decoder_layers=args.dec_layers, 
                         dim_ffn=args.dim_feedforward, num_queries=args.num_queries, update_track_query_pos=args.update_track_pos,
                         enc_use_temporal_attn=args.enc_use_temporal_attn,
                         backbone_name=args.backbone, freeze_backbone=freeze_backbone, backbone_dilation=args.dilation,
                         dropout=args.dropout, aux_output=args.no_aux_output)
    
    initialized_param_names = []
    if args.init_state: # True or class <str>
        path = None if isinstance(args.init_state, bool) else args.init_state
        model, initialized_param_names = load_states_from_pretrained_detr(model, path, num_enc=args.enc_layers, num_dec=args.dec_layers, 
                                                                          load_backbone_state=True, skip_mismatch=True)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ram = n_params * 4 / 1024 ** 2
    print("num of trainable params: {:.2f}M({:.1f}MB)".format(n_params / 1e6, ram))

    param_dicts = [
        {"params": [
            p for n, p in model_without_ddp.named_parameters()
            if "backbone" not in n and n not in initialized_param_names and p.requires_grad
        ]},
    ]

    if not freeze_backbone:
        param_dicts.append({
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        })
    
    if len(initialized_param_names) > 0: # partially initialized
        param_dicts.append({
            "params": [p for n, p in model_without_ddp.named_parameters() if n in initialized_param_names],
            "lr": args.lr * 0.1
        })
    
    # loss
    loss_fn = CollectiveLoss(use_focal, args.focal_alpha, args.cls_coef, args.bbox_coef, args.giou_coef, args.obj_coef)
    
    # optimizer
    optim = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, args.lr_drop, gamma=.1)

    # dataset, sampler & dataloader FIXME: should be instantiated from src.dataset.__init__
    # dataset_train = TAOAmodalDataset(args.root, args.anno_root, args.num_frames,
    #                                  args.sample_interval, mode="train", transform=build_tao_transform())
    # dataset_val = TAOAmodalDataset(args.root, args.anno_root, args.num_frames, 
    #                                mode="val", transform=build_tao_transform())
    if args.dataset_file == "mot17":
        dataset_train, dataset_val = [build_mot_dataset(mode, args) for mode in ["train", "val"]]
    elif args.dataset_file == 'tao':
        raise NotImplementedError()
    else:
        dataset_train = PseudoDataSet(args.num_iters[0] + args.num_frames - 1, num_classes, args.num_frames)
        dataset_val = PseudoDataSet(args.num_iters[0] + args.num_frames - 1, num_classes, args.num_frames)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    dataloader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=collate_fn_tao, num_workers=args.num_workers)
    dataloader_val = DataLoader(dataset_val, sampler=sampler_val, drop_last=False, 
                                 collate_fn=collate_fn_tao, num_workers=args.num_workers)
    
    # init trainer
    trainer = Trainer(model, loss_fn, optim, lr_scheduler,
                      args.clip_max_norm, args.fp_ratio, args.fp_max_score, args.random_drop,
                      # eval params FIXME: to argument parser
                      max_disappear_times=4, max_track_history=10, 
                      cls_conf=0.8, dur_occ_conf=0.5, matching_thres=0.5,
                      log_dir=args.log_dir, device=device)
    
    if args.ckpt_path:
        trainer.load_checkpoint(args.ckpt_path)

    trainer.train(
        dataloader_train, dataloader_val, args.epochs,
        eval_freq=args.eval_freq, save_freq=args.save_freq, save_dir=args.save_dir
    )

    # NOTE: force to save the last checkpoint
    # comment the codes below if you do not want to save the last checkpoint.
    if not args.save_dir:
        save_dir = "./checkpoints"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
    else:
        save_dir = args.save_dir

    trainer.save_checkpoint(
        os.path.join(save_dir, f"{model_without_ddp.name}_epoch_{trainer.current_epoch}.pth")
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training & evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)