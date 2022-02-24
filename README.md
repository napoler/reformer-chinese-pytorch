# reformer-chinese-pytorch
reformer-pytorch中文版本，简单高效的生成模型。类似GPT2的效果

kaggle 示例
https://www.kaggle.com/terrychanorg/reformer-gpt2-otebookf97351bab2

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









# 参数修改示例 jupyter


```python 

%%writefile my_config_test_cuda_16g.yaml

seed_everything: 288
trainer:
  logger:
  - class_path: pytorch_lightning.loggers.WandbLogger
    init_args:
      log_model: false
      offline: false
      prefix: ''
      project: "litGPT"
  checkpoint_callback: true
  callbacks:
  - class_path: pytorch_lightning.callbacks.EarlyStopping
    init_args:
      check_finite: true
      check_on_train_epoch_end: false
      min_delta: 0.0
      mode: min
      monitor: val_loss
      patience: 100
      strict: true
      verbose: true
  - class_path: pytorch_lightning.callbacks.LearningRateMonitor
    init_args:
      log_momentum: false
      logging_interval: step
  - class_path: pytorch_lightning.callbacks.ModelCheckpoint
    init_args:
      auto_insert_metric_name: true
      filename: '{epoch}-{val_loss:.2f}-{val_acc:.2f}'
      mode: min
      monitor: val_loss
      save_last: true
      save_top_k: 2
      save_weights_only: false
      verbose: true
  default_root_dir: null
  gradient_clip_val: 0.0
  gradient_clip_algorithm: norm
  process_position: 0
  num_nodes: 1
  num_processes: 1
  devices: null
  gpus: 1
  auto_select_gpus: false
  tpu_cores: null
  ipus: null
  log_gpu_memory: null
  progress_bar_refresh_rate: null
  overfit_batches: 0.0
  track_grad_norm: -1
  check_val_every_n_epoch: 1
  fast_dev_run: false
  accumulate_grad_batches: 1
  max_epochs: 10
  min_epochs: null
  max_steps: null
  min_steps: null
  max_time: null
  limit_train_batches: 1000
  limit_val_batches: 200
  limit_test_batches: 200
  limit_predict_batches: 1.0
  val_check_interval: 1.0
  flush_logs_every_n_steps: 100
  log_every_n_steps: 50
  accelerator: null
  sync_batchnorm: false
  precision: 16
  weights_summary: top
  weights_save_path: null
  num_sanity_val_steps: 2
  truncated_bptt_steps: null
  resume_from_checkpoint: null
  profiler: null
  benchmark: false
  deterministic: false
  reload_dataloaders_every_n_epochs: 0
  reload_dataloaders_every_epoch: false
  auto_lr_find: false
  replace_sampler_ddp: true
  terminate_on_nan: false
  auto_scale_batch_size: false
  prepare_data_per_node: true
  plugins: null
  amp_backend: native
  amp_level: O2
  distributed_backend: null
  move_metrics_to_cpu: false
  multiple_trainloader_mode: max_size_cycle
  stochastic_weight_avg: false
model:
  dim: 128
  depth: 6
  max_seq_len: 512
  lsh_dropout: 0.1
  optimizer_name: AdamW
  learning_rate: 0.0001
  full_attn_thres: 128
  from_pretrained: bert-base-chinese
  batch_size: 64
  trainfile: ./data/train.pkt
  valfile: ./data/val.pkt
  testfile: ./data/test.pkt


```


Tks：

https://github.com/lucidrains/reformer-pytorch
https://huggingface.co/transformers
https://pytorch-lightning.readthedocs.io/
