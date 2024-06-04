import argparse
import logging
import math
import os
import random
import shutil
import time
from simplex import Simplex
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.kit import get_cosine_schedule_with_warmup,getMaxValue,convert,convert2,deconvert2,entropy_loss, mixup
from Load_Videos import Get_Dataloader,Get_lx_sux_wux_Dataloader_forSI9
from utils import AverageMeter, accuracy
import torch.distributed as dist
from Tool.OTSU import OTSU_threshold
from Tool.SuperAugmentation import Super_Augmentation
from Tool.BalancedDataParallel import BalancedDataParallel as BDP
#from torch.cuda import amp


normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)


logger = logging.getLogger(__name__)
best_acc_a = 0



#Save best epoch
def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True




def main():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--gpu-id', default=0, type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='number of workers')
    parser.add_argument('--dataset', default='Fire', type=str,
                        choices=['cifar10', 'cifar100','ucf101','Fire'],
                        help='dataset name')
    parser.add_argument('--num-classes', default=9, type=int,  #------------cls num--------------
                        help='class num of dataset')
    parser.add_argument("--expand-labels", action="store_true",
                        help="expand labels to fit eval steps")
    #------------------------------------------------------
    parser.add_argument('--arch', default='resnet', type=str,
                        choices=['cnn', 'C3D', 'resnet', 'ResNetV2', 'ResNeXt', 'ResNeXtV2',
                                 'PreActResNet', 'WideResNet', 'DenseNet', 'SqueezeNet', 'ShuffleNetV2',
                                 'ShuffleNet', 'MobileNet', 'MobileNetV2', 'EfficientNet'],
                        help='models name')
    parser.add_argument('--arch2', default='adanet', type=str,
                        help='branch2 name')
    parser.add_argument('--ada_loss_beta', default=0.2, type=float,
                        help='loss_beta')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')

    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold (default=0.95)')
    parser.add_argument('--out', default='result',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=1234, type=int,
                        help="random seed")
    parser.add_argument("--amp", action="store_true",
                        help="use 16-bit (mixed) precision through NVIDIA apex AMP")
    parser.add_argument("--opt_level", type=str, default="O1",
                        help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html")
    #---------------------------------------------------------
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    #----------------------------------------------------------------------------------------
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")
    # -------------------------------for VideoMAE2-------------------------------------------
    parser.add_argument(
        '--drop_path',
        type=float,
        default=0.0,
        metavar='PCT',
        help='Drop path rate (default: 0.1)')
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--tubelet_size', type=int, default=2)
    parser.add_argument(
        '--decoder_depth', default=4, type=int, help='depth of decoder')
    parser.add_argument(
        '--with_checkpoint', action='store_true', default=False)
    parser.add_argument(
        '--mask_ratio', default=0.9, type=float, help='mask ratio of encoder')
    parser.add_argument(
        '--list-threshold', default=0.9, type=float, help='threshold for synthetic queue threshold')

    #====================================================================================================================
    parser.add_argument('--use_ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--input_size', default=160, type=int, help='images input size for backbone')
    parser.add_argument('--num-labeled', type=int, default=80,
                        help='number of labeled data New=8')
    parser.add_argument('--total-steps', default=20000, type=int,
                        help='number of total steps to run 2**20')
    parser.add_argument('--eval-step', default=40, type=int,
                        help='number of eval steps to run 768')
    parser.add_argument('--batch-size', default=2, type=int,
                        help='train batchsize')
    parser.add_argument('--mu', default=2.5, type=float,
                        help='coefficient of unlabeled batch size  New=4')
    parser.add_argument('--embed-dim', default=384, type=int,
                        help='768-384')

    # ====================================================================================================================
    args = parser.parse_args()

    global best_acc_f
    global best_acc_a







   #----------------------------------------------------------------------------
    def create_model(args):

        from all_model.mae2.videomae2 import VisionTransformer as vit
        model = vit(num_classes=args.num_classes, embed_dim=args.embed_dim, img_size=args.input_size)
        #
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,6"
        device_ids = [0,1,2]
        model = torch.nn.DataParallel(model, device_ids=device_ids)


        logger.info("Total model params: {:.2f}M".format(
            sum(p.numel() for p in model.parameters()) / 1e6))
        return model


    #-----------------------------------------------------------------------------------------



    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='gloo')  #linux backend='nccl'   windows='gloo'
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1

    args.device = device

    global best_class_acc
    best_class_acc = []


    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}",)

    logger.info(dict(args._get_kwargs()))

    if args.seed is not None:
        set_seed(args)

    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)
        args.writer = SummaryWriter(args.out)



    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()




    datapath = ['./dataset/SI9t/Train']
    test_datapath = ['./dataset/SI9t/Test']
    weak_augmenta_datapath = ['./dataset/SI9t/Train']
    strong_augmenta_datapath = ['./dataset/SI9t/Train']


    
    test_dataloader, label_dict = Get_Dataloader(test_datapath, 'test', 1)
    class_dict = label_dict


    labeled_train_dataloader, unlabeled_train_dataloader, weak_dataloader, strong_dataloader = \
        Get_lx_sux_wux_Dataloader_forSI9(args,
                                  datapath,
                                  weak_augmenta_datapath,
                                  strong_augmenta_datapath,
                                  'train',
                                  args.batch_size)


    model = create_model(args)


    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)
    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)

    args.epochs = math.ceil(args.total_steps / args.eval_step)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.total_steps)
        
    if args.use_ema:
        from all_model.old.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)

    args.start_epoch = 0

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(
        f"  Total train batch size = {args.batch_size*args.world_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")

    #weights = torch.load('./Pretrained/model_best_224_full_labels.pth', map_location=args.device)
    #model.load_state_dict(weights['model'], strict=False)

    train(args, labeled_train_dataloader, unlabeled_train_dataloader, strong_dataloader, weak_dataloader, test_dataloader,
          model, optimizer, ema_model, scheduler, class_dict)





def train(args,labeled_train_dataloader, unlabeled_train_dataloader, strong_dataloader,weak_dataloader,test_loader,
          model,optimizer, ema_model, scheduler, class_dict):
    if args.amp:
        from torch.cuda import amp
    global best_acc_f
    global best_acc_a
    global best_acc5_a
    global best_avg_acc
    best_avg_acc = 0
    best_acc5_a = 0
    test_accs = []
    end = time.time()
    OTSU_dict = {}
    history_dict = {}
    



    # test_loss, test_acc_f, test_acc_a = test(args, test_loader, model, 1)




    global ema_lamda
    global t
    global local_p
    ema_lamda = args.ema_decay

    classnum = args.num_classes
    default_threshold = 1 / classnum
    t = 0
    local_p = []

    p_model = torch.ones(classnum)
    label_hist = torch.ones(classnum)
    p_model = p_model.to(args.device)
    label_hist = label_hist.to(args.device)



    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        weak_epoch = 0
        strong_epoch = 0
        labeled_train_dataloader.sampler.set_epoch(labeled_epoch)
        strong_dataloader.sampler.set_epoch(weak_epoch)
        weak_dataloader.sampler.set_epoch(strong_epoch)



    model.train()

    #class
    acc_dict = {}
    for epoch in range(args.start_epoch, args.epochs):
        torch.cuda.empty_cache()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_Lx = AverageMeter()
        losses_Lu = AverageMeter()
        losses_ada = AverageMeter()
        mask_probs = AverageMeter()

        a = 0
        b = 0
        c = 0
        d = 0
        e = 0


        labeled_iter = iter(labeled_train_dataloader)
        unlabeled_iter = iter(unlabeled_train_dataloader)

        strong_iter = iter(strong_dataloader)
        weak_iter = iter(weak_dataloader)



        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step),
                         disable=args.local_rank not in [-1, 0])
        
        t_u = []
        t_u_x = []
        for batch_idx in range(args.eval_step):

            try:
                inputs_x, targets_x, _ = labeled_iter.__next__()
            except:
                labeled_iter = iter(labeled_train_dataloader)
                inputs_x, targets_x, _ = labeled_iter.__next__()


            try:
                inputs_u_x, targets_u_x, index_u = unlabeled_iter.__next__()
            except:
                unlabeled_iter = iter(unlabeled_train_dataloader)
                inputs_u_x, targets_u_x, index_u = unlabeled_iter.__next__()

            try:
                inputs_u_w, _, _ = weak_iter.__next__()
                inputs_u_s, _, _ = strong_iter.__next__()
            except:
                strong_iter = iter(strong_dataloader)
                weak_iter = iter(weak_dataloader)
                inputs_u_w, _, _ = weak_iter.__next__() 
                inputs_u_s, _, _ = strong_iter.__next__()


            index_u = index_u.tolist()
            for value in index_u:
                if value in history_dict:
                    OTSU_dict[value] = OTSU_threshold(history_dict[value])
                else:
                    history_dict[value] = []
            for idx, value in enumerate(index_u):
                if len(history_dict[value])!=0 and history_dict[value][-1] < OTSU_dict[value]:
                    inputs_u_s[idx] = Super_Augmentation(inputs_u_s[idx].clone().detach())








            inputs_x = inputs_x.to(args.device)
            inputs_u_x = inputs_u_x.to(args.device)
            targets_x = targets_x.to(args.device)
            
            targets_u_x = targets_u_x.to(args.device)
            
            inputs_u_w = inputs_u_w.to(args.device)
            inputs_u_s = inputs_u_s.to(args.device)


            data_time.update(time.time() - end)


            #---------------------------
            inputs_x = inputs_x.to(torch.float32)     #float32
            inputs_u_w = inputs_u_w.to(torch.float32)
            inputs_u_s = inputs_u_s.to(torch.float32)
            inputs_u_x = inputs_u_x.to(torch.float32)

            num_lb = inputs_x.shape[0]
            inputs = torch.cat((inputs_x, inputs_u_w, inputs_u_s))


            logits, _  = model(inputs)
            logits_x = logits[:num_lb]
            logits_u_w, logits_u_s = logits[num_lb:].chunk(2)





            #----------------------------pseudo_label_list-----------------------------------

            inputs_x_ulb = inputs_u_x
            inputs_y_ulb = torch.softmax(logits_u_w.detach()/args.T, dim=-1)
            inputs_x_ulb = inputs_x_ulb.cpu().clone().numpy()
            inputs_y_ulb = inputs_y_ulb.cpu().clone().numpy()
            indices = np.where(inputs_y_ulb[:, 0] > args.list_threshold)[0]

            list_x_ulb = inputs_x_ulb[indices]
            list_y_ulb = inputs_y_ulb[indices]
            list_x_ulb = torch.tensor(list_x_ulb).to(args.device)
            list_y_ulb = torch.tensor(list_y_ulb).to(args.device)

            list_x_ulb = torch.cat((list_x_ulb,inputs_x))
            max_indices = torch.argmax(list_y_ulb, dim=1)
            labels = 1 - max_indices
            # print(labels.shape[0])
            list_y_ulb = torch.cat((labels,targets_x))

            # --------------------------------------------
            pseudo_label = torch.softmax(logits_u_w.detach() / args.T, dim=-1)

            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            if t == 0:
                global_threshold = default_threshold
                for _ in range(classnum):
                    local_p.append(default_threshold)
                max_local_p = getMaxValue(local_p)
                MaxNorm_p = [x / max_local_p for x in local_p]
                final_threshold = [x * global_threshold for x in MaxNorm_p]
                t = t + 1
            else:
                qbc = pseudo_label.tolist()
                g_threshold = max_probs.tolist()
                total_maxqb = sum(g_threshold)
                global_threshold = ema_lamda + (1 - ema_lamda) * (1 / max_probs.shape[0]) * total_maxqb

                for i in range(classnum):
                    pt = ema_lamda * local_p[i] + (1 - ema_lamda) * (1 / max_probs.shape[0]) * qbc[0][i]
                    local_p[i] = pt

                max_local_p = getMaxValue(local_p)
                MaxNorm_p = [x / max_local_p for x in local_p]
                final_threshold = [x * global_threshold for x in MaxNorm_p]
                t = t + 1
                probs_x_ulb = pseudo_label
                p_model = p_model * ema_lamda + (1 - ema_lamda) * probs_x_ulb.mean(dim=0)
                max_idx = targets_u
                hist = torch.bincount(max_idx.reshape(-1), minlength=p_model.shape[0]).to(p_model.dtype)
                label_hist = label_hist * ema_lamda + (1 - ema_lamda) * (hist / hist.sum())

            f_threshold = []

            for i in targets_u:
                f_threshold.append(final_threshold[i.int()])
            threshold = torch.tensor(f_threshold)
            threshold = threshold.to(args.device)

            mask = torch.ge(max_probs, threshold).float()  

            t_u += targets_u.tolist()
            t_u_x += targets_u_x.tolist()


            in_lx2, x1, x2 = convert2(list_x_ulb)
            in_ux2, _, _ = convert2(inputs_u_x)

            label_onehot = torch.nn.functional.one_hot(list_y_ulb.to(torch.int64),
                                                       args.num_classes).squeeze().float().to(args.device)

            logits_u_x,_ = model(inputs_u_x)


            # -----------------------------------------------------------

            # -----------------------------------------------------------------------------
            mixup_video, mixup_label, mix_indice = mixup(
                in_lx2,  # (9,96,160,160)*4
                label_onehot,  # (2,2) [0., 1.]
                in_ux2,  # (24,96,160,160)
                logits_u_x, # (8,2) [0.9362, 0.0638]
                inputs_x.shape[0],
                args
            )
            mixup_label = F.softmax(mixup_label, dim=1)


            mixup_video = deconvert2(mixup_video,x1,x2)
            # ----------------------------------------
            mixup_pred, mixup_cls = model(mixup_video)


            # -----------------------------------------------------------


            assert Simplex(mixup_pred) and Simplex(mixup_cls)


            ada_reg_loss = F.cross_entropy(mixup_pred, mixup_label, reduction='mean')
            ada_adv_loss = F.cross_entropy(mixup_cls, mix_indice, reduction='mean')

            # ada_total_loss = (ada_reg_loss + ada_adv_loss) * 0.1 + ada_labeled_loss * 0.2
            ada_total_loss = ada_reg_loss + ada_adv_loss

            Lx = F.cross_entropy(logits_x, targets_x.long().cuda(), reduction='mean')



            update_dict = (F.cross_entropy(logits_u_s.clone().detach(), targets_u.clone().detach(),
                                  reduction='none'))
            update_dict = update_dict.tolist()
            flag = 0
            for index in index_u:
                history_dict[index].append(update_dict[flag])
                flag +=1
            flag = 0


            Lu = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()

            if mask.sum() > 0:
               ent_loss, _ = entropy_loss(mask, logits_u_s, p_model, label_hist)
            else:
               ent_loss = 0.0
               
        
            loss = Lx + Lu + ada_total_loss + 0.01 * ent_loss
            
            a = a + Lx.item()
            b = b + Lu.item()
            e = e + loss.item()
            c = c + ada_total_loss.item()





            loss.backward()

            losses.update(loss.item())
            losses_Lx.update(Lx.item())
            losses_Lu.update(Lu.item())
            losses_ada.update(ada_total_loss.item())
            # losses_f.update(ent_loss.item())
            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()
            batch_time.update(time.time() - end)
            end = time.time()
            mask_probs.update(mask.float().mean().item())
            if not args.no_progress:
                p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_Lx: {loss_Lx:.4f}. Loss_Lu: {loss_Lu:.4f}. Loss_ada: {loss_ada:.4f}. Mask: {mask:.2f}.".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=args.eval_step,
                    lr=scheduler.get_last_lr()[0],
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    loss_Lx=losses_Lx.avg,
                    loss_Lu=losses_Lu.avg,
                    loss_ada=losses_ada.avg,
                    mask = mask_probs.avg))
                p_bar.update()
                
        correct_predictions = sum(u == x for u, x in zip(t_u, t_u_x))
        total_predictions = len(t_u)
        accuracy = correct_predictions / total_predictions
        acc_dict[epoch] = accuracy


        if not args.no_progress:
            p_bar.close()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        if args.local_rank in [-1, 0]:
            # test_model = test_model.to(args.device).eval()
            test_loss, test_acc_a, test_acc5_a, class_acc_list, class_num_list = test(args, test_loader, test_model, False)

            args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
            args.writer.add_scalar('train/2.train_loss_x', losses_Lx.avg, epoch)
            args.writer.add_scalar('train/3.train_loss_u', losses_Lu.avg, epoch)
            args.writer.add_scalar('train/3.train_loss_ada', losses_ada.avg, epoch)
            args.writer.add_scalar('train/4.mask', mask_probs.avg, epoch)
            # args.writer.add_scalar('test/1.test_f_acc', test_acc_f, epoch)
            args.writer.add_scalar('test/2.test_a_acc', test_acc_a, epoch)
            args.writer.add_scalar('test/3.test_loss', test_loss, epoch)

            avg_acc = 0.0
            
            for i in range(len(class_acc_list)):
                avg_acc += (class_acc_list[i] * 100) / args.num_classes
  

            is_best = (test_acc_a >= best_acc_a)
            is_avg_best = avg_acc >= best_avg_acc
            
            if is_best:
                best_acc_a = max(test_acc_a, best_acc_a)
                best_acc5_a = max(test_acc5_a, best_acc5_a)
                if is_avg_best:
                    best_avg_acc = avg_acc
                    best_class_acc = class_acc_list

                    for i in range(len(best_class_acc)):
                        #print("Class:",class_dict[int(i)], "acc=",(best_class_acc[i] * 100), "num=", class_num_list[i])
                        print("Class: {:<20} acc={:<7.2f} num={:<7}".format(class_dict[int(i)], best_class_acc[i] * 100, class_num_list[i]))
                print("Best average acc=", best_avg_acc)
                


            model_to_save = model.module if hasattr(model, "module") else model

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'acc': test_acc_a,
                'best_acc_a': best_acc_a,
                'best_acc5_a': best_acc5_a,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_avg_best, args.out)

            test_accs.append(test_acc_a)

            logger.info('Best_a top-1 acc: {:.2f}'.format(best_acc_a))
            logger.info('Best_a top-5 acc: {:.2f}'.format(best_acc5_a))

    

    file_path = "SIAVC_pl_acc.txt"
    with open(file_path, "w") as file:
        for key, value in acc_dict.items():
            file.write(f"{key}: {value}\n")
    print("Dictionary saved to", file_path)
    # loss_data.to_excel('loss.xlsx', index=False)
    if args.local_rank in [-1, 0]:
        args.writer.close()


def test(args, test_loader, model, Test):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1_f = AverageMeter()
    top5_f = AverageMeter()
    top1_a = AverageMeter()
    top5_a = AverageMeter()
    end = time.time()

    t_SNE = False
    features =[]
    labels = []
    class_num = [0.0] * args.num_classes
    class_acc_num = [0.0] * args.num_classes
    result_list = [] * args.num_classes
    result_num_list = [0.0] * args.num_classes





    if Test :
        weights = torch.load('./Pretrained/SIAVC.pth.tar', map_location=args.device)
        model.load_state_dict(weights['state_dict'], strict=True)
        t_SNE = False
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        f = []
        l = []





    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0])

    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()


            inputs = inputs.to(args.device)  # float16
            inputs = inputs.to(torch.float32)
            targets = targets.to(args.device)  # float32
            targets = targets.to(torch.int64)


            outputs2, _  = model(inputs)


            if t_SNE:
                features_batch = outputs2.cpu().numpy()
                features.append(features_batch)
                labels.append(targets.cpu().numpy())


            prec1_a, prec5_a = accuracy(outputs2, targets, topk=(1,5))

            _, predicted = outputs2.max(1)

            for i in range(len(targets)):
                class_num[targets[i]] += 1.0

                index = predicted.tolist()

                if index[i] == (targets[i]):
                    class_acc_num[index[i]] += 1.0





            top1_a.update(prec1_a.item(), inputs.shape[0])
            top5_a.update(prec5_a.item(), inputs.shape[0])



            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description("Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1_a: {top1_a:.2f}. top5_a: {top5_a:.2f}.".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1_a=top1_a.avg,
                    top5_a=top5_a.avg,
                ))

        for i in range(len(class_num)):
            result = class_acc_num[i] / class_num[i]
            result_list.append(result)
            result_num = class_acc_num[i]




        if not args.no_progress:
            test_loader.close()



    logger.info("top-1-a acc: {:.2f}".format(top1_a.avg))
    logger.info("top-5-a acc: {:.2f}".format(top5_a.avg))
    return losses.avg, top1_a.avg, top5_a.avg, result_list, class_acc_num



if __name__ == '__main__':
    main()

