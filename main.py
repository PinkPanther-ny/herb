from src.loss import LossSelector
from src.optim import OptSelector
from src.models import ModelSelector
from src.preprocess import Preprocessor
from src.settings import configs
from src.utils import Timer, eval_total, remove_bad_models, set_random_seeds

import torch.distributed as dist
import datetime
import torch
from torch.optim.lr_scheduler import MultiStepLR
import gc

# OMP_NUM_THREADS=2 python -m torch.distributed.run --nproc_per_node 4 main.py

def train():
    set_random_seeds()
    if configs._LOCAL_RANK == 0:
        print(f"\n==================  Loading required configurations  ==================\n")
    # DDP backend initialization
    if configs.DDP_ON:
        torch.cuda.set_device(configs._LOCAL_RANK)
        dist.init_process_group(backend='nccl')

    
    # Define model, loss function and optimizer for the following training process
    model = ModelSelector(configs).get_model()
    criterion = LossSelector(loss_name=configs.LOSS, cfg=configs).get_loss()
    optimizer = OptSelector(model.parameters(), opt_name=configs.OPT, cfg=configs).get_optim()
    scheduler = MultiStepLR(optimizer, milestones=configs.LEARNING_RATE_DECREASE_EPOCHS, gamma=configs.LEARNING_RATE_GAMMA)
    
    trainloader, testloader = Preprocessor().get_loader()
    
    # Start timer from here
    timer = Timer()
    timer.timeit()
    
    if configs._LOCAL_RANK == 0:
        print(f"\n================== ================================= ==================\n")
        if configs._LOAD_SUCCESS:
            print(f"Verifying loaded model ({configs.MODEL_NAME.replace('/','')})'s accuracy as its name suggested...")
            # eval_total(model, testloader, timer)
        print(f"\n================== Start training! Total {configs.TOTAL_EPOCHS} epochs ==================\n")
    
    # Mixed precision for massive speed up
    # https://zhuanlan.zhihu.com/p/165152789
    scalar = None
    if configs.MIX_PRECISION:
        scalar = torch.cuda.amp.GradScaler()
    
    # ========================== Train =============================
    for epoch in range(configs.TOTAL_EPOCHS):
        
        # Just for removing bad models
        remove_bad_models()
        if configs.DDP_ON:
            # To avoid duplicated data sent to multi-gpu
            trainloader.sampler.set_epoch(epoch)
        
        # Counter for printing information during training
        count_log = 0 if configs.N_LOGS_PER_EPOCH == 0 else int(len(trainloader) / configs.N_LOGS_PER_EPOCH)
        
        for i, data in enumerate(trainloader, 0):
            t = timer.timeit()
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

            if (configs.LOG_EVERY_TIME and configs._LOCAL_RANK == 0) or (count_log != 0 and configs._LOCAL_RANK == 0 and i % count_log == count_log - 1):
                print(f'[{epoch + 1}(Epochs), {i + 1:5d}/{len(trainloader)}(batches)], Already: {t[1]}\n')

        # Count epochs for learning rate scheduler
        scheduler.step()
        
        # Evaluate model on main GPU after EPOCHS_PER_EVAL epochs
        if configs._LOCAL_RANK == 0:
            if scheduler.get_last_lr() != scheduler.get_lr():
                print(f"Learning rate updated from {scheduler.get_last_lr()} to {scheduler.get_lr()}")
            # Time current epoch training duration
            t = timer.timeit()
            if epoch % configs.EPOCHS_PER_EVAL == configs.EPOCHS_PER_EVAL - 1:
                eval_total(model, testloader, timer, epoch)
            print(f"Epoch delta time: {t[0]}, Already: {t[1]}\n")

    print(f'Training Finished! ({str(datetime.timedelta(seconds=int(timer.timeit())))})')


if __name__ == '__main__':
    try:
        gc.collect()
        torch.cuda.empty_cache()
        train()
    except KeyboardInterrupt:
        print("Exit!")

