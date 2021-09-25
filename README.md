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
