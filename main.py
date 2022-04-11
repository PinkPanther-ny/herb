import torch
import torch.distributed as dist
import gc
import os
from tqdm import tqdm

from src.settings import configs, logger
from src.loss import LossSelector
from src.models import ModelSelector
from src.optim import OptSelector, SkdSelector
from src.preprocess import Preprocessor
from src.utils import set_random_seeds, Timer, eval_total, gen_submission, remove_bad_models, save_checkpoint

import torch
from torch.utils.tensorboard import SummaryWriter

# OMP_NUM_THREADS=2 python -m torch.distributed.run --nproc_per_node 4 main.py

def train():
    if configs.TENSOR_BOARD_ON:
        writer = SummaryWriter()
    set_random_seeds()
    if configs._LOCAL_RANK == 0:
        logger.info(f"==================  Loading required configurations  ==================\n")
    # DDP backend initialization
    if configs.DDP_ON:
        torch.cuda.set_device(configs._LOCAL_RANK)
        dist.init_process_group(backend='nccl')

    # Define model, loss function and optimizer for the following training process
    model = ModelSelector().get_model()
    criterion = LossSelector().get_loss()
    optimizer = OptSelector(model.parameters()).get_optim()
    scheduler = SkdSelector(optimizer).get_skd()

    p = Preprocessor()
    train_loader, test_loader = p.get_loader()

    # Start timer from here
    timer = Timer()
    timer.timeit()

    if configs._LOCAL_RANK == 0:
        logger.info(f"[Train | Test] of batches [({int(os.environ['WORLD_SIZE'])}(gpus) * {len(train_loader)}) | ({len(test_loader)})] with batch size: {configs.BATCH_SIZE} loaded!")
        logger.info(f"Total {len(train_loader.dataset)+len(test_loader.dataset)} data points!\n")
        logger.info(f"================== ================================= ==================\n")
        if configs._LOAD_SUCCESS:
            logger.info(f"Verifying loaded model ({configs.MODEL_NAME.replace('/', '')})'s accuracy as its name suggested...")
            eval_total(model, test_loader)

            if configs.GEN_SUBMISSION:
                logger.info(f"Generating submission file")
                gen_submission(model, p.get_submission_test_loader())
        else:
            logger.info('No model loaded, skip verifying step\n')
        logger.info(f"================== Start training! Total {configs.TOTAL_EPOCHS} epochs ==================\n")

    model.train()
    # Mixed precision for massive speed up
    # https://zhuanlan.zhihu.com/p/165152789
    scalar = None
    if configs.MIX_PRECISION:
        scalar = torch.cuda.amp.GradScaler()

    # ========================== Train =============================
    for epoch in range(configs._CUR_EPOCHS, configs.TOTAL_EPOCHS + 1):
        timer.timeit()
        # Just for removing bad models
        remove_bad_models()
        if configs.DDP_ON:
            # To avoid duplicated data sent to multi-gpu
            train_loader.sampler.set_epoch(epoch)

        # Disable tqdm bar if current rank is not 0
        p_bar = tqdm(train_loader, ncols=160, colour='blue', unit='batches', disable=configs._LOCAL_RANK!=0)

        epoch_loss = 0
        for i, data in enumerate(p_bar, 1):
            if configs._LOCAL_RANK == 0:
                desc = f'Epoch {epoch} batch {i}'
                p_bar.set_description(f"{desc:18s}")

            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # Speed up with half precision
            if configs.MIX_PRECISION:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs.to(configs._DEVICE))
                    loss = criterion(outputs, labels.to(configs._DEVICE))

                    # Scale the gradient
                    scalar.scale(loss).backward()
                    scalar.step(optimizer)
                    scalar.update()
            else:
                outputs = model(inputs.to(configs._DEVICE))
                loss = criterion(outputs, labels.to(configs._DEVICE))
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
            if configs._LOCAL_RANK == 0:
                p_bar.set_postfix({'loss': f"{round(epoch_loss/i, 4)}", 'lr': optimizer.param_groups[0]['lr']})
        # Count epochs for learning rate scheduler
        scheduler.step()

        # Evaluate model on main GPU after EPOCHS_PER_EVAL epochs
        if configs._LOCAL_RANK == 0:
            # Time current epoch training duration
            if epoch % configs.EPOCHS_PER_EVAL == configs.EPOCHS_PER_EVAL - 1:
                acc = eval_total(model, test_loader, epoch)
                
                if configs.TENSOR_BOARD_ON:
                    writer.add_scalars("Accuracy|Loss",  {'accuracy': acc,
                                                        'loss': epoch_loss/len(train_loader)}, epoch)
                save_checkpoint(model, acc, optimizer, scheduler, epoch)
                
        
        train_loss = round(epoch_loss/len(train_loader), 4)
        cur_lr = round(optimizer.param_groups[0]['lr'], 8)
        
        if configs._LOCAL_RANK == 0 and configs.TENSOR_BOARD_ON:
            writer.add_scalar("LR/train", cur_lr, epoch)
        
        logger.info(f"Epoch {epoch}: training_loss = {train_loss}, learning_rate = {cur_lr}")
        t = timer.timeit()
        logger.info(f"Epoch delta time: {t[0]}, Already: {t[1]}\n")
    
    logger.info(f"Training Finished! ({timer.timeit()[1]})\n")


if __name__ == '__main__':
    try:
        gc.collect()
        torch.cuda.empty_cache()
        train()
    except KeyboardInterrupt:
        logger.warning(f"Training stopped by KeyboardInterrupt!\n")
        logger.info("Exit!")
