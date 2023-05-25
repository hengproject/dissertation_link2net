import torch

from util.save_util import save_some_example
from util.test_util import _cat_and_unsqueeze, PSNR, SSIM, MS_SSIM, _L1Loss, _criterionMSE


def _get_evaluate_indication(img_list_a:[torch.Tensor],img_list_b:[torch.Tensor]):
    a_tensor = _cat_and_unsqueeze(img_list_a)
    b_tensor = _cat_and_unsqueeze(img_list_b)

    sum_PSNR = 0
    length = len(img_list_a)
    for i in range(length):
        sum_PSNR += PSNR(img_list_a[i],img_list_b[i])
    avg_PSNR = sum_PSNR / length
    ssim = SSIM(a_tensor,b_tensor,data_range=1).item()
    ms_ssim = MS_SSIM(a_tensor,b_tensor,data_range=1).item()
    mae = _L1Loss(a_tensor,b_tensor).item()
    mse = _criterionMSE(a_tensor,b_tensor).item()

    return {
        "avg_PSNR": avg_PSNR,
        "ssim":ssim,
        "ms_ssim":ms_ssim,
        "mae": mae,
        "mse": mse
    }


def get_evaluate_indication(G,test_data_loader,device):
    with torch.no_grad():
        G.eval()
        result: [torch.Tensor] = save_some_example(num=-1, gen=G, test_dataloader=test_data_loader,
                                                   save=False, ret_result=True, device=device)
        b_list = []
        c_list = []

        for each in result:
            a, b, c = torch.chunk(each, 3, dim=2)
            b_list.append(b)
            c_list.append(c)
            # c_list.append(a)
        G.train()
        return _get_evaluate_indication(b_list, c_list)