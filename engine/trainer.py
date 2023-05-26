import numpy as np
import torch.cuda
import matplotlib.pyplot as plt
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage
import torch.nn as nn
from utils.reid_metric import R1_mAP
from loguru import logger

'''
ignite是一个高级的封装训练和测试库
'''


global ITER
ITER = 0
# 重写create_supervised_trainer，规则是在内部实现一个def _update(engine, batch),最后返回Engine(_update)
def create_supervised_trainer(model, optimizer, loss_fn, device=None, kd=False, feature_loss_coefficient=0.03):
    """
    :param model:  (nn.Module) reid model to train
    :param optimizer:Adam or SGD
    :param loss_fn: loss function
    :param device: gpu or cpu
    :return: Engine
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        img, target = batch
        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        target = target.to(device) if torch.cuda.device_count() >= 1 else target
        if not kd:
            score, feat = model(img) # 采用表征+度量
            loss = loss_fn(score, feat, target)  # 传入三个值，score是fc层后的(hard)，feat是池化后的特征，target是标签
            loss.backward()
            optimizer.step()
            # compute acc
            acc = (score.max(1)[1] == target).float().mean()
            return loss.item(), acc.item()
        elif kd:
            score, feat, layer_out_feat = model(img)
            loss = loss_fn(score, feat, target)
            teacher_feature = layer_out_feat[1].detach()  # 取出教师层
            '''
            （rannge(idx,len(layer_out_feat))，中的idx可以决定哪个作为教师）
            idx=1表示layer4为教师网络,layer3,layer2,layer1为student
            idx=2表示layer3为教师网络,layer2,layer1为student
            idx=3表示layer2为教师网络，layer1为student
            '''
            for index in range(2, len(layer_out_feat)):  # layer4, layer3, layer2, layer1
                if index != 2:  # 排除自己
                    loss += torch.dist(layer_out_feat[index], teacher_feature) * feature_loss_coefficient
            loss.backward()
            optimizer.step()
            # compute acc
            acc = (score.max(1)[1] == target).float().mean()
            return loss.item(), acc.item()
    return Engine(_update)


# 重写create_supervised_evaluator,传入model和metrics,metrics是一个字典用来存储需要度量的指标
def create_supervised_evaluator(model, metrics, device=None):
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pids, camids = batch
            data = data.to(device) if torch.cuda.is_available() else data
            feat = model(data)
            return feat, pids, camids
    engine = Engine(_inference)
    for name, metric in metrics.items():
        metric.attach(engine, name)
    return engine


def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        num_query,
        start_epoch,
        kd,
        feature_loss_coefficient
):
    log_period = 1
    checkpoint_period = 1
    eval_period = 1
    output_dir = cfg.OUTPUT_DIR
    device = cfg.DEVICE
    epochs = cfg.MAX_EPOCHS
    print("Start training~")
    trainer = create_supervised_trainer(model, optimizer, loss_fn, device, kd, feature_loss_coefficient)
    evaluator = create_supervised_evaluator(model,
                                            metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm='yes')},
                                            device=device)
    checkpointer = ModelCheckpoint(output_dir, cfg.model_name, checkpoint_period, n_saved=10, require_empty=False)
    state_dict = model.state_dict()
    timer = Timer(average=True)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model,
                                                                     'optimizer': optimizer})
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)
    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc')


    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        global ITER
        ITER += 1

        if ITER % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                        .format(engine.state.epoch, ITER, len(train_loader),
                                engine.state.metrics['avg_loss'], engine.state.metrics['avg_acc'],
                                scheduler.get_lr()[0]))
        if len(train_loader) == ITER:
            ITER = 0

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        logger.info('准备测试阶段，请稍等....')
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:
            evaluator.run(val_loader)
            cmc, mAP = evaluator.state.metrics['r1_mAP']
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            text = "mAP:{:.1%}".format(mAP)
            # logger.info("mAP: {:.1%}".format(mAP))
            logger.info(text)
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
            if not cfg.kd:
                torch.save(state_dict, 'logs/mAP_{:.1%}.pth'.format(mAP))
            else:
                torch.save(state_dict, 'logs/mAP_KD_{:.1%}.pth'.format(mAP))
            return cmc, mAP

    trainer.run(train_loader, max_epochs=epochs)

