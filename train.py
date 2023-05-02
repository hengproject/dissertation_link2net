import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from copy import deepcopy
from network.loss import GANLoss, MixedIndicationLoss
from network.public import get_schedulers, update_learning_rate
from util import logger_util, visualize_util
from util.time_util import time_start, time_end, time_end_as_minutes
from util.save_util import init_save, check_point_save
from util.load_util import yaml_plain_loader as config_loader
from util.load_util import load_checkpoint
from data.ect_dataset import get_train_dataset, get_test_dataset
from network.network_getter import define_G, define_D
import argparse


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml',type=str,help='the path of config.yml',default='./config.yml')
    args = parser.parse_args()
    config = config_loader(args.yaml)

    train_opt = config['train_opt']
    dataset_opt = config['dataset_opt']
    save_opt = train_opt['save_opt']

    save_dir = init_save(save_opt['root_path'], config)

    log = logger_util.init_logger(save_dir)
    log.info(f"config are {config}")
    log.info(f"cuda is available : {torch.cuda.is_available()}")

    if config['use_cuda'] and not torch.cuda.is_available():
        log.fatal("cuda is not available, set use_cuda option to false")
        exit(1)

    device = None
    if config['use_cuda']:
        device = 'cuda:0'
    else:
        device = 'cpu'

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    if train_opt['seed']['set']:
        torch.manual_seed(train_opt['seed']['num'])
        log.debug(f"seed is set as {train_opt['seed']['num']} manually")

    ram_cached = dataset_opt['ram_cache']
    log.info(f'===> Loading datasets process start, ram_cached = {ram_cached}')
    timer = time_start()
    root_path = dataset_opt['root_path']
    dataset_name = dataset_opt['dataset_name']
    direction = dataset_opt['direction']
    use_aug = dataset_opt['use_aug']
    log.debug(f"root_path is {root_path}/{dataset_name}")
    # 返回一个数组 0 为 目标图像 1 为输入图像
    train_dataset = get_train_dataset(root_path, dataset_name, direction, ram_cached=ram_cached,use_augment=use_aug)
    test_dataset = get_test_dataset(root_path, dataset_name, direction, ram_cached=False)
    log.info(f'===> Loading datasets process done, using {time_end(timer)} ms ; {time_end_as_minutes(timer)} minutes')
    num_workers = train_opt['num_workers']
    batch_size = train_opt['batch_size']
    # after this, data input = a , output = b
    training_data_loader = DataLoader(dataset=train_dataset, num_workers=num_workers, batch_size=batch_size,
                                      shuffle=True)
    testing_data_loader = DataLoader(dataset=test_dataset, num_workers=num_workers, batch_size=1,
                                     shuffle=False)
    log.info('===> Building models process start')

    input_channel = train_opt['channel']['input']
    output_channel = train_opt['channel']['output']
    ngf = train_opt['G']['ngf']
    net_G_type = train_opt['G']['net_G_type']
    net_G_init_type = train_opt['G']['G_init_type']
    net_G_use_dropout = train_opt['G']['use_dropout']
    # input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02,
    # gpu_id='cuda:0'
    net_generator = define_G(input_channel, output_channel, ngf, net_G_type, norm='batch',
                             use_dropout=net_G_use_dropout,
                             init_type=net_G_init_type, init_gain=0.02, gpu_id=device)

    ndf = train_opt['D']['ndf']
    net_D_type = train_opt['D']['net_D_type']
    net_D_init_type = train_opt['D']['D_init_type']
    # input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_id='cuda:0'
    # input_nc = real_a.channel + fake_b.channel
    net_discriminator = define_D(input_channel + output_channel, ndf, net_D_type, init_type=net_D_init_type,
                                 gpu_id=device)
    log.debug(f'generator : {net_generator}')
    log.debug(f'discriminator : {net_discriminator}')

    loss_mode = train_opt['loss']['mode']
    criterionGAN = GANLoss(gan_mode=loss_mode).to(device)
    criterionGLoss = MixedIndicationLoss(train_opt['loss']['g_loss']).to(device)
    criterionL1 = torch.nn.L1Loss().to(device)

    # optimizer
    g_lr = float(train_opt['optim']['g_learning_rate'])
    d_lr = float(train_opt['optim']['d_learning_rate'])
    betas = (
        float(train_opt['optim']['betas']['beta1']), float(train_opt['optim']['betas']['beta2']))
    # optim.Adam(net_generator.parameters(), lr=lr, betas=betas)
    # optim.Adam(net_discriminator.parameters(), lr=lr, betas=betas)

    # torch.optim.lr_scheduler provides several methods to adjust the learning rate based on the number of epochs.
    optimizer_g = optim.Adam(params=net_generator.parameters(), lr=g_lr, betas=betas)
    optimizer_d = optim.Adam(params=net_discriminator.parameters(), lr=d_lr, betas=betas)

    if train_opt['continue_train_opt']['activate']:
        continue_train_opt = train_opt['continue_train_opt']
        load_checkpoint(continue_train_opt['g_model_path'],net_generator,optimizer_g,device)
        load_checkpoint(continue_train_opt['d_model_path'],net_discriminator,optimizer_d,device)


    # last epoch will be inherited here
    net_g_schedulers = get_schedulers(optimizer_g, config)
    net_d_schedulers = get_schedulers(optimizer_d, config)

    start_epoch = int(train_opt['starting_epoch'])
    end_epoch = start_epoch + int(train_opt['train_n_epochs'])

    log.info("training start")
    for epoch in range(start_epoch, end_epoch):
        training_data_loader_len = len(training_data_loader)
        epoch_timer = time_start()
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        # train
        for iteration, batch in enumerate(training_data_loader, start=0):
            iter_timer = time_start()
            # 1. forward
            # get real a,b batch from dataset
            real_a, real_b = batch[0].to(device), batch[1].to(device)
            fake_b = net_generator(real_a)
            # 2. update D network
            optimizer_d.zero_grad()

            # 2.1 train with fake (get loss_d_fake)
            # real_a and fake_b
            fake_ab = torch.cat((real_a, fake_b), dim=1)
            # detach fake_ab from current graph to avoid auto gradient
            prediction_fake = net_discriminator.forward(fake_ab.detach())
            loss_d_fake = criterionGAN(prediction=prediction_fake, target_is_real=False)
            # 2.2 train with real (get loss_d_real)
            # real_a and real_b
            real_ab = torch.cat((real_a, real_b), dim=1)
            # detach real_ab from current graph to avoid auto gradient
            # in original, no detach() is applied
            prediction_real = net_discriminator.forward(real_ab)
            # loss_d_real = nn.MSE() or nn.BCELoss() prediction should be real
            loss_d_real = criterionGAN(prediction=prediction_real, target_is_real=True)
            # 2.3 combine d_loss
            loss_d = (loss_d_fake + loss_d_real) * 0.5
            # 2.4 update
            loss_d.backward()
            optimizer_d.step()

            # 3. update generator network
            optimizer_g.zero_grad()

            # 3.1 G(a_real) try to fake D , use criterionGAN to assess to what degree D consider G(a_real) is true
            fake_ab = torch.cat((real_a, fake_b), 1)
            prediction_fake = net_discriminator.forward(fake_ab)
            loss_g_from_gan = criterionGAN(prediction=prediction_fake, target_is_real=True)

            # 3.2 Creates a criterion that measures the mean absolute error (MAE) between each element in the input x and target y
            loss_g_from_l1 = criterionL1(fake_b, real_b) * float(train_opt['l1']['weight'])
            loss_g = loss_g_from_gan + loss_g_from_l1
            loss_g.backward()
            optimizer_g.step()
            # end of batch
            loss_d_item = loss_d.item()
            loss_g_item = loss_g.item()
            epoch_d_loss += loss_d_item
            epoch_g_loss += loss_g_item
            log.debug(
                f"===> Epoch[{epoch}]({iteration}/{training_data_loader_len}): Loss_D:{round(loss_d_item, 5)} loss_G:{round(loss_g_item, 5)}  using {time_end(iter_timer)} ms")
        # end of epoch
        g_lr = update_learning_rate(net_g_schedulers, optimizer_g, round(epoch_g_loss / training_data_loader_len, 5))
        d_lr = update_learning_rate(net_d_schedulers, optimizer_d, round(epoch_d_loss / training_data_loader_len, 5))
        log.info(
            f"===> Epoch[{epoch}]: using {time_end_as_minutes(epoch_timer)} minutes, avg_Loss_D: {round(epoch_d_loss / training_data_loader_len, 5)}, avg_Loss_G: {round(epoch_g_loss / training_data_loader_len, 5)}")
        log.info(
            f"===> Epoch[{epoch}]: learning rate for next epoch：G_learning_rate:{g_lr},D_learning_rate:{d_lr}"
        )
        if save_opt['save_model']['activate'] and epoch % int(save_opt['save_model']['frequency']) == 0 and epoch != 0:
            temp_config = deepcopy(config)

            check_point_save(save_folder=save_dir, epoch=epoch, gen=net_generator, disc=net_discriminator,
                             gen_optim=optimizer_g, disc_optim=optimizer_d, g_lr=g_lr, d_lr=d_lr, save_opt=save_opt,
                             test_dataloader=testing_data_loader, config=temp_config,device=device
                             )


if __name__ == '__main__':
    run()
