from util.evaluate_util import get_evaluate_indication
from util.logger_util import get_logger


def tensorboard_save(writer,  epoch,  loss_dict):
    writer.add_scalars(main_tag='train_indication', tag_scalar_dict=loss_dict, global_step=epoch)
    get_logger().info(f'epoch[[{epoch}]] tensorboard saved')
