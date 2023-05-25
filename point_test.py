from os.path import join as path_join
from os import listdir as ls

from torch.utils.data import DataLoader

from data.ect_dataset import get_test_dataset
from network.network_getter import define_G
from util.evaluate_util import get_evaluate_indication
from util.load_util import yaml_plain_loader as config_loader, load_checkpoint_only_net
from util.save_util import _init_dir, save_some_example


def run(info_list, savepath, define_test_dataset_path):
    device = 'cuda'
    for info in info_list:
        path = info['path']
        print(path)
        to_test_epochs = info['epochs']
        save_dir = path_join(path)
        config = config_loader(path_join(save_dir, 'train_config.yaml'))
        dataset_opt = config['dataset_opt']
        train_opt = config['train_opt']
        _init_dir(savepath)
        epoch_list = [epoch for epoch in ls(save_dir) if epoch.isnumeric()]
        epoch_list = [int(epoch) for epoch in epoch_list]
        dataloader = test_dataloader(dataset_opt, define_test_dataset_path)
        G = getG(train_opt, device)
        net_G_type = train_opt['G']['net_G_type']
        for epoch in to_test_epochs:
            if epoch not in epoch_list:
                continue
            model_save_path = path_join(save_dir, str(epoch), 'gen_model.pth')
            result_save_path = path_join(savepath, net_G_type, str(epoch))
            load_checkpoint_only_net(model_save_path, G, device)
            # save_some_example(-1, G, dataloader, device, result_save_path)
            # print(f'save_done')
            ret = get_evaluate_indication(G, dataloader, device)
            print(net_G_type, epoch, ret)


def test_dataloader(dataset_opt, define_path=None):
    root_path = dataset_opt['root_path']
    dataset_name = dataset_opt['dataset_name']
    if define_path is not None:
        root_path = define_path
    direction = dataset_opt['direction']
    test_dataset = get_test_dataset(root_path, dataset_name, direction, ram_cached=False)
    test_data_loader = DataLoader(dataset=test_dataset, num_workers=2, batch_size=1,
                                  shuffle=False, )
    return test_data_loader


def getG(train_opt, device):
    input_channel = train_opt['channel']['input']
    output_channel = train_opt['channel']['output']
    ngf = train_opt['G']['ngf']
    net_G_type = train_opt['G']['net_G_type']
    G = define_G(input_channel, output_channel, ngf, net_G_type, norm='batch', gpu_id=device)
    return G


if __name__ == "__main__":
    infos = [
        {
            'path': r'F:\save_data\link2net\link2net50_1m_8s_6block_several\2023-05-16--12-09-28',
            'epochs': [0,200]
        },
        {
            'path': r'F:\save_data\link2net\link2net50_1m_8s_6block_several\2023-05-16--13-14-48',
            'epochs': [0, 200]
        },
        {
            'path': r'F:\save_data\link2net\link2net50_1m_8s_6block_several\2023-05-16--14-20-10',
            'epochs': [0, 200]
        },
        {
            'path': r'F:\save_data\link2net\link2net50_1m_8s_6block_several\2023-05-16--15-26-02',
            'epochs': [0, 200]
        },
        {
            'path': r'F:\save_data\link2net\link2net50_1m_8s_6block_several\2023-05-16--16-31-44',
            'epochs': [0, 200]
        },

    ]
    savepath = path_join(r'F:\零时存储\可以删除\dissertation_save', '1')
    define_test_dataset_path = r'B:\pix2pix'
    run(infos, savepath, define_test_dataset_path)
