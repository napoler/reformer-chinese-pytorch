# -*- coding: utf-8 -*-
import os
import torch
from torch import nn
import torch.nn.functional as F
# from torchvision import transforms
# from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import torch
# from torch import randint
from reformer_pytorch import ReformerLM
from reformer_pytorch.generative_tools import TrainingWrapper
from transformers import BertTokenizer
from tkitLr import CyclicCosineDecayLR

class LitGPT(pl.LightningModule):
    def __init__(self, dim=128, depth=6, max_seq_len=512, lsh_dropout=0.1, optimizer_name="AdamW",learning_rate=1e-4,full_attn_thres=128, from_pretrained='bert-base-chinese', batch_size=2, trainfile="./data/train.pkt", valfile="./data/val.pkt", testfile="./data/test.pkt", **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = BertTokenizer.from_pretrained(from_pretrained)

        model = ReformerLM(
            num_tokens=self.tokenizer.vocab_size,
            dim=dim,
            depth=depth,
            max_seq_len=max_seq_len,
            lsh_dropout=lsh_dropout,
            causal=True,
            full_attn_thres=full_attn_thres
        )

        # 0 is used for padding and no loss to be calculated on it
        self.model = TrainingWrapper(
            model, ignore_index=self.tokenizer.pad_token_id, pad_value=self.tokenizer.pad_token_id)
        print(self.hparams)

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        # embedding = self.encoder(x)
        loss = self.model(x, return_loss=True)
        return loss

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    #     return optimizer
    
    def configure_optimizers(self):
        """优化器 自动优化器"""
        optimizer = getattr(torch.optim, self.hparams.optimizer_name)(self.parameters(), lr=self.hparams.learning_rate)
        #         使用自适应调整模型
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=500,factor=0.8,verbose=True)
        scheduler = CyclicCosineDecayLR(optimizer, 
                                        init_decay_epochs=10,
                                        min_decay_lr=1e-6,
                                        restart_interval = 5,
                                        restart_lr=self.hparams.learning_rate/2,
                                        restart_interval_multiplier=1.5,
                                        warmup_epochs=20,
                                        warmup_start_lr=self.hparams.learning_rate/0.2)
#
        lr_scheduler={
            'scheduler': scheduler,
            'interval': 'epoch',
            'frequency': 1,
            'name':"lr_scheduler",
            'monitor': 'train_loss', #监听数据变化
            'strict': True,
        }
#         return [optimizer], [lr_scheduler]
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler} 
    def training_step(self, batch, batch_idx):
        x = batch
        loss = self(x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        loss = self(x)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x = batch
        loss = self(x)
        self.log('test_loss', loss)
        return loss

    def train_dataloader(self):
        train = torch.load(self.hparams.trainfile)
        return DataLoader(train, batch_size=int(self.hparams.batch_size), num_workers=2, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        val = torch.load(self.hparams.valfile)
        return DataLoader(val, batch_size=int(self.hparams.batch_size), num_workers=2, pin_memory=True)

    def test_dataloader(self):
        val = torch.load(self.hparams.testfile)
        return DataLoader(val, batch_size=int(self.hparams.batch_size), num_workers=2, pin_memory=True)
