import torch

from torch import nn

from torch.cuda.amp import autocast, GradScaler

from torch.utils.data import DataLoader

from loader import *



from engine import *

import os

import sys



os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # "0, 1, 2, 3"



from utils import *

from configs.config_setting import setting_config



import warnings



warnings.filterwarnings("ignore")





# torch.backends.cudnn.deterministic = True



import os

import sys

import argparse

import importlib.util





from utils import *

import warnings



warnings.filterwarnings("ignore")



# 从配置文件加载默认参数

def load_config(config_path):

    spec = importlib.util.spec_from_file_location("config", config_path)

    config = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(config)

    return config





# 解析命令行参数

def parse_args():

    parser = argparse.ArgumentParser(description="Training script")



    # 配置文件参数：现在默认指向 `configs/config_setting_CGNet_DySample_GSConvns_CBAM.py`

    parser.add_argument('--config', type=str, default='configs/config_setting.py', help='Path to the config file (Python script)')



    # 数据集参数：如果没有传递，使用配置文件中的默认值

    parser.add_argument('--dataset', type=str, help='Dataset name')



    # 训练轮数参数：如果没有传递，使用配置文件中的默认值

    parser.add_argument('--epochs', type=int, help='Number of epochs for training')

    parser.add_argument('--batch_size', type=int, help='Number of batch_size for training')



    parser.add_argument('--network', type=str, help='network')

    parser.add_argument('--M', type=int, help='M')

    parser.add_argument('--N', type=int, help='N')



    return parser.parse_args()





def main():

    # 解析命令行参数

    args = parse_args()



    # 打印命令行参数，确保解析工作正常

    print(f"Command line arguments: {args}")





    # 加载配置文件

    config = load_config(args.config)

    if args.network is not None:

        config.setting_config.network = args.network

        from datetime import datetime

        config.setting_config.work_dir = 'results/' + args.network + str(args.batch_size)+ '_' + args.dataset  + '_' + datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss') + '/'



    if args.M is not None:

        config.setting_config.M =   int(args.M )

        print(config.setting_config.M,'------------------------')

    if args.N is not None:

        config.setting_config.N = int(args.N )

        print(config.setting_config.N,'------------------------')



    # 覆盖配置文件中的数据集参数

    if args.dataset is not None:

        # 直接覆盖配置中的 dataset 设置，确保命令行优先

        config.setting_config.datasets = args.dataset



        print('config.setting_config.datasets ',config.setting_config.datasets )

        if args.dataset == 'ISIC2017':
            config.setting_config.data_path = 'data/dataset_isic17/'

        elif args.dataset == 'ISIC2018':
            config.setting_config.data_path = 'data/dataset_isic18/'

        elif args.dataset == 'ISIC2016':
            config.setting_config.data_path = 'data/ISIC2016/'

        elif args.dataset == 'PH2':

            config.setting_config.data_path = 'data/PH2/'

        elif args.dataset == 'DRIVE':

            config.setting_config.data_path = 'data/DRIVE/'

        elif args.dataset == 'MoNuSeg':

            config.setting_config.data_path = 'data/MoNuSeg/'

        elif args.dataset == 'CVC_ClinicDB':

            config.setting_config.data_path = 'data/CVC_ClinicDB/'

        elif args.dataset == 'Glas':

            config.setting_config.data_path = 'data/Glas/'

        elif args.dataset == 'plug':

            config.setting_config.data_path = 'data/plug/'

        elif args.dataset == 'huatielu':

            config.setting_config.data_path = 'data/huatielu/'
        elif args.dataset == 'Kvasir-SEG':

            config.setting_config.data_path = 'data/Kvasir-SEG/'


        elif args.dataset == "FUSC2021":
            config.setting_config.data_path = "data/FUSC2021/"


        elif args.dataset == "CVC-ClinicDB":
            config.setting_config.data_path = "data/CVC-ClinicDB/"
        elif args.dataset == "Kvasir":
            config.setting_config.data_path = "data/Kvasir/"
        elif args.dataset == "CVC-ColonDB":  #训练集没掺和，这是验证泛化能力的
            config.setting_config.data_path = "data/CVC-ColonDB/"
        elif args.dataset == "EndoScene": #训练集没掺和，这是验证泛化能力的
            config.setting_config.data_path = "data/EndoScene/"
        elif args.dataset == "ETIS": #训练集没掺和，这是验证泛化能力的
            config.setting_config.data_path = "data/ETIS/"

        elif args.dataset == "isic18_Glas": #训练集没掺和，这是验证泛化能力的
            config.setting_config.data_path = "data/isic18_Glas/"
        elif args.dataset == "REFUGE2": #训练集没掺和，这是验证泛化能力的
            config.setting_config.data_path = "data/REFUGE2/"
        elif args.dataset == "conic": #训练集没掺和，这是验证泛化能力的
            config.setting_config.data_path = "data/conic/"
        elif args.dataset == "MMOTU":
            config.setting_config.data_path = "data/MMOTU/"





        else:

            raise ValueError("Unsupported dataset!")

    else:

        # 如果命令行没有传入 datasets，则使用默认设置

        print(f"Using default dataset from setting_config: {config.setting_config.datasets}")





    # 覆盖 epochs 参数，如果命令行没有传递，则使用配置文件中的默认值

    if args.epochs is not None:

        config.setting_config.epochs = args.epochs

    else:

        print(f"Using default epochs: {config.setting_config.epochs}")



    # 覆盖 batch_size 参数，如果命令行没有传递，则使用配置文件中的默认值

    if args.batch_size is not None:

        config.setting_config.batch_size = args.batch_size

    else:

        print(f"Using default epochs: {config.setting_config.batch_size}")









    # 输出确认

    print(f"Final config: dataset={config.setting_config.datasets}, data_path={config.setting_config.data_path}")

    print(f"epochs={config.setting_config.epochs}, batch_size={config.setting_config.batch_size}")



    # 继续调用训练流程

    train(config.setting_config)





def train(config):

    print('#----------Creating logger----------#')

    # sys.path.append(config.work_dir + '/')

    sys.path.append(config.work_dir + '/')



    log_dir = os.path.join(config.work_dir, 'log')

    print(log_dir,config.work_dir,'------------------------------------------')

    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')

    resume_model = os.path.join(checkpoint_dir, 'latest.pth')

    outputs = os.path.join(config.work_dir, 'outputs')

    if not os.path.exists(checkpoint_dir):

        os.makedirs(checkpoint_dir)

    if not os.path.exists(outputs):

        os.makedirs(outputs)



    global logger

    logger = get_logger('train', log_dir)



    log_config_info(config, logger)



    print('#----------GPU init----------#')

    set_seed(config.seed)

    gpu_ids = [0]  # [0, 1, 2, 3]

    torch.cuda.empty_cache()



    print('#----------Preparing dataset----------#')

    train_dataset = isic_loader(path_Data=config.data_path, train=True)
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=config.num_workers)
    val_dataset = isic_loader(path_Data=config.data_path, train=False)
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=config.num_workers,
                            drop_last=True)

    test_dataset = isic_loader(path_Data=config.data_path, train=False, Test=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             pin_memory=True,
                             num_workers=config.num_workers,
                             drop_last=True)



    print('#----------Prepareing Models----------#')
    model_cfg = config.model_config

    from get_model import get_model
    # network =
    model = get_model(config.network)



    #-------------------补充消融实验，逐个模块添加-------------------------------

     # model = torch.nn.DataParallel(model.cuda(), device_ids=gpu_ids, output_device=gpu_ids)

    model = torch.nn.DataParallel(model.cuda(), device_ids=[0], output_device=gpu_ids[0])

# device_id列表，显示多卡训练    output_device默认卡0

    print('#----------Prepareing loss, opt, sch and amp----------#')

    criterion = config.criterion

    optimizer = get_optimizer(config, model)

    scheduler = get_scheduler(config, optimizer)

    scaler = GradScaler()



    print('#----------Set other params----------#')

    min_loss = float('inf')

    start_epoch = 1

    min_epoch = 1



    if os.path.exists(resume_model):

        print('#----------Resume Model and Other params----------#')

        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))

        model.module.load_state_dict(checkpoint['model_state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        saved_epoch = checkpoint['epoch']

        start_epoch += saved_epoch

        min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']



        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'

        logger.info(log_info)



    # 创建权重保存目录
    weights_dir = 'weights'
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    
    # 最佳权重文件路径
    best_weight_path = os.path.join(weights_dir, f'{config.network}_{config.datasets}_best.pth')
    
    # 记录全局历史最优值（用于比较，不会被当前训练覆盖）
    global_best_loss = float('inf')
    
    # 加载已有的最优权重（如果存在）
    if os.path.exists(best_weight_path):
        print(f'#----------加载历史最优权重 {best_weight_path}----------#')
        try:
            # 加载权重到模型
            model_dict = model.module.state_dict()
            pretrained_dict = torch.load(best_weight_path, map_location='cpu')
            
            # 读取历史最优损失（如果存在记录）
            if isinstance(pretrained_dict, dict) and 'loss' in pretrained_dict:
                global_best_loss = pretrained_dict['loss']
                print(f'历史最优模型损失值: {global_best_loss:.6f}')
            
            # 提取权重状态
            weight_dict = pretrained_dict
            if isinstance(pretrained_dict, dict) and 'state_dict' in pretrained_dict:
                weight_dict = pretrained_dict['state_dict']
            elif isinstance(pretrained_dict, dict) and 'model_state_dict' in pretrained_dict:
                weight_dict = pretrained_dict['model_state_dict']
                
            # 处理权重字典
            if isinstance(weight_dict, dict):
                # 如果是字典格式（state_dict）
                matched_dict = {k: v for k, v in weight_dict.items() 
                               if k in model_dict and v.shape == model_dict[k].shape}
                
                model_dict.update(matched_dict)
                model.module.load_state_dict(model_dict)
                logger.info(f'成功加载历史最优权重，共{len(matched_dict)}/{len(model_dict)}个参数')
            else:
                # 如果是直接的state_dict
                model.module.load_state_dict(weight_dict)
                logger.info(f'成功加载历史最优权重文件')
                
            print(f'已加载历史最优权重，将在此基础上继续训练')
        except Exception as e:
            logger.error(f'加载历史最优权重失败: {str(e)}')
            print(f'加载历史最优权重出错: {str(e)}，将忽略历史权重')
    else:
        print(f'未发现历史最优权重，将从头开始训练')
    
    # 设置当前训练的最优损失为局部最优
    local_min_loss = min_loss
    
    for epoch in range(start_epoch, config.epochs + 1):
        torch.cuda.empty_cache()
        train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            logger,
            config,
            scaler=scaler
        )



        loss = val_one_epoch(

            val_loader,

            model,

            criterion,

            epoch,

            logger,

            config

        )



        # 在训练循环内修改保存逻辑
        if loss < local_min_loss:
            local_min_loss = loss
            
            # 只有当前模型超过全局最优时才保存
            if loss < global_best_loss:
                global_best_loss = loss
                # 同时保存损失值以便未来比较
                torch.save({
                    'model_state_dict': model.module.state_dict(),
                    'epoch': epoch,
                    'loss': loss
                }, best_weight_path)
                print(f'发现新的全局最优模型! 保存到：{best_weight_path}')
                logger.info(f'已保存新的全局最优权重，epoch={epoch}，loss={loss:.6f}')
            else:
                print(f'当前模型是局部最优但未超过历史全局最优 ({loss:.6f} vs {global_best_loss:.6f})')
                logger.info(f'当前模型未超过历史最优，不更新全局权重')



        # torch.save(
        #
        #     {
        #
        #         'epoch': epoch,
        #
        #         'min_loss': local_min_loss,
        #
        #         'min_epoch': min_epoch,
        #
        #         'loss': loss,
        #
        #         'model_state_dict': model.module.state_dict(),
        #
        #         'optimizer_state_dict': optimizer.state_dict(),
        #
        #         'scheduler_state_dict': scheduler.state_dict(),
        #
        #     }, os.path.join(checkpoint_dir, 'latest.pth'))



    # if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
    if os.path.exists(best_weight_path):

        print('#----------Testing----------#')

        # 安全加载最佳权重
        print(f'#----------加载测试权重 {best_weight_path}----------#')
        try:
            checkpoint = torch.load(best_weight_path, map_location='cpu')
            
            # 处理多GPU参数前缀
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            if not isinstance(model, nn.DataParallel) and any(k.startswith('module.') for k in state_dict):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            elif isinstance(model, nn.DataParallel) and not any(k.startswith('module.') for k in state_dict):
                state_dict = {'module.'+k: v for k, v in state_dict.items()}
            
            # 过滤不匹配参数
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() 
                              if k in model_dict and v.shape == model_dict[k].shape}
            
            # 加载参数
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=False)
            
            # 记录加载信息
            logger.info(f'成功加载测试权重: {len(pretrained_dict)}/{len(model_dict)} 参数')
            missing = [k for k in model_dict if k not in pretrained_dict]
            if missing:
                logger.warning(f'缺失参数: {missing}')
            unexpected = [k for k in pretrained_dict if k not in model_dict]
            if unexpected:
                logger.warning(f'冗余参数: {unexpected}')

        except Exception as e:
            logger.error(f'加载测试权重失败: {str(e)}')
            raise

        # 执行测试
        print('#----------开始正式测试----------#')
        model.eval()
        
        loss = test_one_epoch(
            test_loader,
            model,
            criterion,
            logger,
            config,
        )
        
        # # 将单值loss转换为字典格式
        # if not isinstance(loss, dict):
        #     loss = {'loss': loss}
        #
        # # 保存测试结果
        # result_file = os.path.join(config.work_dir, 'test_results.txt')
        # with open(result_file, 'w') as f:
        #     f.write(f"测试数据集: {config.datasets}\n")
        #     f.write(f"最佳权重路径: {best_weight_path}\n")
        #     f.write("测试指标:\n")
        #     for k, v in loss.items():
        #         f.write(f"{k}: {v:.4f}\n")
        #
        # print(f'#----------测试完成，结果保存至 {result_file}----------#')


# if __name__ == '__main__':

#     config = setting_config

#     main(config)



if __name__ == '__main__':

    main()
