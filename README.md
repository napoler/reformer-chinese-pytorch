# reformer-chinese-pytorch
reformer-pytorch中文版本，简单高效的生成模型。类似GPT2的效果


# 数据

需要训练的数据文件放置在 data/data.txt
纯文本文件，一条一行即可

# 预处理数据

> python bulidData.py


# 训练

# Dump default configuration to have as reference
导出配置文件

> python trainer.py  --print_config > config/default_config.yaml

# Modify the config to your liking - you can remove all default arguments

nano config.yaml
# Fit your model using the configuration
运行训练

> python trainer.py --config  config/default_config.yaml




# 其他

查看cuda占用

> watch -n 1 nvidia-smi









# 参数修改示例


```python 

import yaml

with open("default_config.yaml", 'r') as stream:
    try:
#         print(yaml.safe_load(stream))
        conf=yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
# conf["model"]["batch_size"]=32
    
    conf["model"]["batch_size"]=1024
    conf["model"]["learning_rate"]=3e-3
    
    conf["model"]["out_num_classes"]=37
    
    conf["seed_everything"]=42
    conf["trainer"]["max_epochs"]=2000
    conf["trainer"]["gpus"]=-1
    conf["trainer"]["precision"]=16

    
    conf["trainer"]["checkpoint_callback"]=True
    conf["trainer"]["logger"]=[{'class_path': 'pytorch_lightning.loggers.WandbLogger', 'init_args': {'offline': False, 'project': 'tkit-tagger chip2020—notebook751089c5ee', 'log_model': False, 'prefix': ''}}]
    conf["trainer"]["callbacks"]= [{'class_path': 'pytorch_lightning.callbacks.EarlyStopping', 
                                    'init_args': {'monitor': 'val_loss', 'min_delta': 0.0, 'patience': 100, 
                                                  'verbose': True, 'mode': 'min', 'strict': True, 'check_finite': True, 'check_on_train_epoch_end': False}}, 
                                   {'class_path': 'pytorch_lightning.callbacks.LearningRateMonitor', 'init_args': {'logging_interval': 'step', 'log_momentum': False}}, 
                                   {'class_path': 'pytorch_lightning.callbacks.ModelCheckpoint', 'init_args': {'filename': '{epoch}-{val_loss:.2f}-{val_acc:.2f}', 'monitor': 'val_loss', 'verbose': True, 'save_last': True, 'save_top_k': 2, 'save_weights_only': False, 'mode': 'min', 'auto_insert_metric_name': True}}]
    
#     conf["trainer"]["logger"]["project"]="tkit-taggerchip2020—notebook751089c5ee"
#     conf["trainer"]["logger"]["project"]="tkit-taggerchip2020—notebook751089c5ee"


    # conf["model"]["batch_size"]=3

    with open("default_config.yaml", 'w') as stream:
        print(conf)  
        yaml.dump(conf, stream)

```