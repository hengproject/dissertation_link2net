import argparse
import os.path
from os.path import join as path_join
from os import listdir as ls
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data.ect_dataset import get_test_dataset
from util.load_util import yaml_plain_loader as config_loader, load_checkpoint_only_net
from util.save_util import init_test_save
from util.load_util import get_device
from network.network_getter import define_G
from util.evaluate_util import get_evaluate_indication


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, help='the path of train_save dir', required=True)
    parser.add_argument('--save_example', type=bool, help='save example or not', default=False)
    args = parser.parse_args()
    save_dir = path_join(args.save_path)
    save_example = args.save_example
    config = config_loader(path_join(save_dir, 'train_config.yaml'))
    device = get_device(config)
    device = 'cpu'

    dataset_opt = config['dataset_opt']
    train_opt = config['train_opt']
    test_save_dir = init_test_save(save_dir)
    epoch_list = [epoch for epoch in ls(save_dir) if epoch.isnumeric()]
    epoch_list = [int(epoch) for epoch in epoch_list]
    epoch_list.sort()

    root_path = dataset_opt['root_path']
    dataset_name = dataset_opt['dataset_name']
    direction = dataset_opt['direction']
    test_dataset = get_test_dataset(root_path, dataset_name, direction, ram_cached=False)
    test_data_loader = DataLoader(dataset=test_dataset, num_workers=2, batch_size=1,
                                  shuffle=False)

    input_channel = train_opt['channel']['input']
    output_channel = train_opt['channel']['output']
    ngf = train_opt['G']['ngf']
    net_G_type = train_opt['G']['net_G_type']
    G = define_G(input_channel, output_channel, ngf, net_G_type, norm='batch', gpu_id=device)
    writer = SummaryWriter(os.path.join(save_dir,'test','logs'), flush_secs=1)
    for epoch in epoch_list:
        model_save_path = path_join(save_dir, str(epoch), 'gen_model.pth')
        result_save_path = path_join(test_save_dir, str(epoch))
        load_checkpoint_only_net(model_save_path, G, device)
        ret = get_evaluate_indication(G,test_data_loader,device)
        print(epoch,ret)
        writer.add_scalars(main_tag='test_indication', tag_scalar_dict=ret, global_step=epoch)

    writer.close()



if __name__ == "__main__":
    run()
