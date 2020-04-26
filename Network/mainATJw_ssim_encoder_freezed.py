import importlib
import argparse, os, time
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import DatasetFromHdf5
import utils
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from Vgg16 import Vgg16
import pytorch_msssim
# Training settings
parser = argparse.ArgumentParser(description="Pytorch DRRN")
parser.add_argument("--batchSize", type=int, default=7, help="Training batch size")
parser.add_argument("--patchSize", type=int, default=256, help="Training patch size")
parser.add_argument("--traindata", default="/mnt/ramdisk/256_training_2020.h5", type=str, help="Training datapath")#SynthesizedfromN18_256s64

parser.add_argument("--nEpochs", type=int, default=150, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate, Default=0.1")
parser.add_argument("--step", type=int, default=10, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default=5")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--aug", action="store_true", help="Use aug?")

# parser.add_argument("--resume", default="model/dense_residual_deepModel_AT_actual_finetune/model_epoch_4.pth", type=str, help="Path to checkpoint, Default=None")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint, Default=None")
parser.add_argument("--start-epoch", default=1, type = int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--clip", type=float, default=0.01, help="Clipping Gradients, Default=0.01")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default=1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default=0.9")
parser.add_argument("--weight_decay", "--wd", default=1e-6, type=float, help="Weight decay, Default=1e-4")
parser.add_argument("--activation", default="no_relu", type=str, help='activation relu')
parser.add_argument("--ID", default="ssim_encoder_freezed", type=str, help='ID for training')
parser.add_argument("--model", default="commondense_separetetrans_dilation_inception_resblocks_clamp", type=str, help="unet or drrn or runet")


def main():
    global opt, model
    opt = parser.parse_args()
    print(opt)

    save_path = os.path.join('.', "model", "{}_{}".format(opt.model, opt.ID))
    log_dir = './records/{}_{}/'.format(opt.model, opt.ID)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    cuda = opt.cuda = True
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    # opt.seed = 4222

    print("Random Seed: ", opt.seed)

    cudnn.benchmark = True

    print("===> Building model")
    try:
        mod = importlib.import_module(opt.model)
        try:
            model = mod.Model()
        except AttributeError:
            model = mod.Dense()
    except FileExistsError:
        raise SyntaxError('wrong model type {}'.format(opt.model))


    # model.freeze('_pretrained')
    model.freeze('encoder')

    print("===> Loading datasets")
    train_set = DatasetFromHdf5(opt.traindata, opt.patchSize, opt.aug)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,
                                      shuffle=True)

    criterion = nn.MSELoss()
    Absloss = nn.L1Loss()
    ssim_loss = pytorch_msssim.MSSSIM()
    #loss_var = torch.std()

    vgg = Vgg16(requires_grad=False)

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("===> loading checkpoint: {}".format(opt.resume))
            model, epoch, dict = utils.load_checkpoint(opt.resume)
            opt.start_epoch = epoch + 1
        else:
            raise FileNotFoundError("===> no checkpoint found at {}".format(opt.resume))

    print("===> Setting GPU")
    if cuda:
        model = torch.nn.DataParallel(model).cuda()
        criterion = criterion.cuda()
        Absloss = Absloss.cuda()
        ssim_loss = ssim_loss.cuda()
        #loss_var = loss_var.cuda()
        vgg = vgg.cuda()

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)#weight_decay=opt.weight_decay

    print("===> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        # utils.save_checkpoint(model, epoch, opt, save_path)
        train(training_data_loader, optimizer, model, criterion, Absloss, ssim_loss, epoch, vgg)
        utils.save_checkpoint(model, epoch, opt, save_path)
        # os.system("python eval.py --cuda --model=model/model_epoch_{}.pth".format(epoch))


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt.lr * (0.5 ** (epoch  // opt.step))
    return lr


def train(training_data_loader, optimizer, model, criterion, Absloss, ssim_loss, epoch, vgg):

    writer = SummaryWriter(log_dir='./records/{}_{}/'.format(opt.model, opt.ID))

    # lr policy
    lr = adjust_learning_rate(optimizer, epoch-1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))

    model.train()
    total_iter = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = Variable(batch[0]), Variable(batch[1])
        if opt.cuda:
            input = input.cuda()
            target = target.cuda()

        # outputs
        J_total, J_direct, J_AT, A, T, w = model(input, opt)

        features_target = vgg(target)
        features_output_eh = vgg(J_total)
        # haze = target*T + (1.0 - T)*A
        loss_mse_total = criterion(J_total, target)
        loss_mse_direct = criterion(J_direct, target)
        loss_mse_AT = criterion(J_AT, target)
        # loss_mse_recon = criterion(haze, input)
        # loss_l1_total = Absloss(J_total, target)
        # loss_l1_direct = Absloss(J_direct, target)
        # loss_l1_AT = Absloss(J_AT, target)

        loss_vgg = criterion(features_output_eh.relu2_2, features_target.relu2_2) + criterion(features_output_eh.relu1_2, features_target.relu1_2) + criterion(features_output_eh.relu3_3, features_target.relu3_3)
        loss_vgg = loss_vgg/3.0
        s_loss = 1.0 - ssim_loss(target, J_total)
        var_loss = torch.std(A)**2
        loss = loss_mse_total + 0.7*(loss_mse_direct + loss_mse_AT) + 0.5*loss_vgg + .01*var_loss + 0.5*s_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #writer.add_scalar('data/scalar1', loss.item(), total_iter)

        if iteration%50 == 0:
            psnr_run_total = utils.PSNR_self(J_total.clone(), target.clone())
            psnr_run_direct = utils.PSNR_self(J_direct.clone(), target.clone())
            psnr_run_AT = utils.PSNR_self(J_AT.clone(), target.clone())

            input_display = vutils.make_grid(input, normalize=False, scale_each=True)
            #writer.add_image('Image/train_input', input_display, total_iter)

            J_total_display = vutils.make_grid(J_total, normalize=False, scale_each=True)
            J_direct_display = vutils.make_grid(J_direct, normalize=False, scale_each=True)
            J_AT_display = vutils.make_grid(J_AT, normalize=False, scale_each=True)
            #writer.add_image('Image/train_output', output_display, total_iter)

            gt_display = vutils.make_grid(target, normalize=False, scale_each=True)
            #writer.add_image('Image/train_target', gt_display, total_iter)

            # psnr_run = 100
            print("===> Epoch[{}]({}/{}): Loss_mse: {:.10f} , Loss_vgg: {:.10f} ,psnr: {:.3f}, psnr_AT: {:.3f}, psnr_direct: {:.3f}"
                  "".format(epoch, iteration, len(training_data_loader), loss_mse_total.data.item(), loss_vgg, psnr_run_total, psnr_run_AT, psnr_run_direct))
        total_iter += 1


if __name__ == "__main__":
    main()
