# src/scvi/model/base/_mlxmixin.py

from __future__ import annotations

import logging
import warnings
from typing import Optional, Union, Dict, Any, List, Callable, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from scvi.dataloaders import DataSplitter
from scvi.model._utils import get_max_epochs_heuristic
from scvi.train import TrainRunner
from scvi.utils._docstrings import devices_dsp

logger = logging.getLogger(__name__)

class MlxTrainingPlan:
    """用于MLX模块的训练计划。
    
    此训练计划处理MLX模型的优化过程，包括前向传递、损失计算和参数更新。
    
    参数
    ----------
    module
        MLX模块实例。
    lr
        学习率。
    weight_decay
        权重衰减系数。
    n_epochs_kl_warmup
        KL散度权重从min_kl_weight扩展到max_kl_weight的轮数。
    n_steps_kl_warmup
        KL散度权重从min_kl_weight扩展到max_kl_weight的步数。
    max_kl_weight
        训练期间KL散度的最大缩放因子。
    min_kl_weight
        训练期间KL散度的最小缩放因子。
    **loss_kwargs
        传递给模型损失方法的额外关键字参数。
    """
    
    def __init__(
        self, 
        module, 
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        n_epochs_kl_warmup: Optional[int] = 400,
        n_steps_kl_warmup: Optional[int] = None,
        max_kl_weight: float = 1.0,
        min_kl_weight: float = 0.0,
        **loss_kwargs
    ):
        self.module = module
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs_kl_warmup = n_epochs_kl_warmup
        self.n_steps_kl_warmup = n_steps_kl_warmup
        self.loss_kwargs = loss_kwargs
        self.max_kl_weight = max_kl_weight
        self.min_kl_weight = min_kl_weight
        
        # 初始化步数和轮数计数器
        self.current_step = 0
        self.current_epoch = 0
        
        # 创建MLX优化器（AdamW包含权重衰减）
        self.optimizer = optim.AdamW(
            learning_rate=self.lr, 
            weight_decay=self.weight_decay,
            eps=1e-8,
            betas=[0.9, 0.999],
            bias_correction=True
        )
    
    def _set_module_training(self, is_training):
        """设置模块的训练状态。
        
        在缺少标准train()和eval()方法的情况下使用。
        
        参数
        ----------
        is_training : bool
            模块是否处于训练模式。
        """
        # 尝试不同的可能属性，以便兼容不同的模块实现
        if hasattr(self.module, "_is_training"):
            self.module._is_training = is_training
        elif hasattr(self.module, "_training"):
            self.module._training = is_training
        else:
            logger.warning("无法设置模块的训练状态，可能影响训练效果")
    
    def get_kl_weight(self) -> float:
        """计算当前步骤或轮次的KL权重。
        
        返回
        -------
        float
            当前KL权重值。
        """
        if self.n_steps_kl_warmup is not None:
            kl_weight = min(
                self.max_kl_weight,
                self.min_kl_weight + (self.max_kl_weight - self.min_kl_weight) * 
                (self.current_step / max(1, self.n_steps_kl_warmup))
            )
        elif self.n_epochs_kl_warmup is not None:
            kl_weight = min(
                self.max_kl_weight,
                self.min_kl_weight + (self.max_kl_weight - self.min_kl_weight) * 
                (self.current_epoch / max(1, self.n_epochs_kl_warmup))
            )
        else:
            kl_weight = self.max_kl_weight
        
        return kl_weight
    
    def train_step(self, batch_dict):
        """执行单步训练。
        
        参数
        ----------
        batch_dict
            包含批次数据的字典。
            
        返回
        -------
        dict
            包含损失值的字典。
        """
        # 设置模块为训练模式
        if hasattr(self.module, "train"):
            self.module.train(True)
        else:
            self._set_module_training(True)
        
        # 将批次数据转换为MLX数组
        mlx_batch = {k: mx.array(v) if isinstance(v, np.ndarray) else v for k, v in batch_dict.items()}
        
        # 计算KL权重
        kl_weight = self.get_kl_weight()
        
        try:
            # 定义损失函数 - 使用MLX风格
            def loss_fn(model, tensors, kl_weight):
                _, _, loss_output = model(tensors, kl_weight=kl_weight)
                return loss_output.loss
            
            # 使用mlx.nn.value_and_grad计算损失和梯度
            loss_and_grad_fn = nn.value_and_grad(self.module, loss_fn)
            # 修复调用方式，显式传递self.module
            loss, grads = loss_and_grad_fn(self.module, mlx_batch, kl_weight)
            
            # 使用优化器直接更新模块参数
            self.optimizer.update(self.module, grads)
            
            # 强制评估参数和优化器状态
            mx.eval(self.module.parameters(), self.optimizer.state)
            
            return {"loss": float(loss)}
        except Exception as e:
            logger.error(f"训练步骤出错: {str(e)}")
            raise
    
    def validate_step(self, batch_dict):
        """执行验证步骤。
        
        参数
        ----------
        batch_dict
            包含批次数据的字典。
            
        返回
        -------
        dict
            包含验证损失值的字典。
        """
        # 设置模块为评估模式
        if hasattr(self.module, "eval"):
            self.module.eval()
        else:
            self._set_module_training(False)
        
        # 将批次数据转换为MLX数组
        mlx_batch = {k: mx.array(v) if isinstance(v, np.ndarray) else v for k, v in batch_dict.items()}
        
        # 计算KL权重（与训练时相同，以保持一致性）
        kl_weight = self.get_kl_weight()
        
        try:
            # 前向传递
            if hasattr(self.module, "__call__"):
                _, _, loss_output = self.module(mlx_batch, kl_weight=kl_weight)
                return {"validation_loss": float(loss_output.loss)}
            else:
                # 兼容模式
                logger.warning("模块没有标准的__call__方法，使用兼容模式")
                return {"validation_loss": 0.0}
        except Exception as e:
            logger.error(f"验证步骤出错: {str(e)}")
            raise


class MlxTrainingMixin:
    """MLX模型的训练混合类。
    
    此混合类为使用MLX作为后端的模型提供训练功能。
    它处理数据分割、训练循环和评估。
    """
    
    _data_splitter_cls = DataSplitter
    _training_plan_cls = MlxTrainingPlan
    _train_runner_cls = TrainRunner
    
    @devices_dsp.dedent
    def train(
        self,
        max_epochs: Optional[int] = None,
        accelerator: str = "auto",
        devices: Union[int, List[int], str] = "auto",
        train_size: Optional[float] = None,
        validation_size: Optional[float] = None,
        shuffle_set_split: bool = True,
        batch_size: int = 128,
        datasplitter_kwargs: Optional[dict] = None,
        plan_kwargs: Optional[dict] = None,
        **trainer_kwargs,
    ):
        """训练模型。
        
        参数
        ----------
        max_epochs
            数据集的通过次数。如果为None，默认为
            `np.min([round((20000 / n_cells) * 400), 400])`
        accelerator
            加速器类型。MLX自动选择最佳可用设备。
        devices
            设备选择。MLX自动选择最佳可用设备。
        train_size
            训练集大小，范围[0.0, 1.0]。
        validation_size
            验证集大小。如果为None，默认为1 - train_size。
        shuffle_set_split
            分割前是否打乱索引。
        batch_size
            训练期间使用的小批量大小。
        datasplitter_kwargs
            DataSplitter的额外关键字参数。
        plan_kwargs
            训练计划的关键字参数。
        **trainer_kwargs
            训练的额外关键字参数。
            
        返回
        -------
        self
            训练后的模型实例。
        """
        if max_epochs is None:
            max_epochs = get_max_epochs_heuristic(self.adata.n_obs)
        
        logger.info(f"使用MLX训练模型，共{max_epochs}轮")
        
        # 创建数据分割器
        datasplitter_kwargs = datasplitter_kwargs or {}
        data_splitter = self._data_splitter_cls(
            self.adata_manager,
            train_size=train_size,
            validation_size=validation_size,
            shuffle_set_split=shuffle_set_split,
            batch_size=batch_size,
            iter_ndarray=True,
            **datasplitter_kwargs,
        )
        
        # 设置数据分割器
        data_splitter.setup()
        
        # 创建训练计划
        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else {}
        self.training_plan = self._training_plan_cls(self.module, **plan_kwargs)
        
        # 获取数据加载器
        train_dl = data_splitter.train_dataloader()
        val_dl = data_splitter.val_dataloader()
        
        # 训练循环
        self.training_plan.current_epoch = 0
        self.training_plan.current_step = 0
        
        for epoch in range(max_epochs):
            self.training_plan.current_epoch = epoch
            epoch_loss = 0.0
            n_batches = 0
            
            # 训练阶段
            for batch in train_dl:
                self.training_plan.current_step += 1
                try:
                    output = self.training_plan.train_step(batch)
                    epoch_loss += output["loss"]
                    n_batches += 1
                except Exception as e:
                    logger.error(f"训练批次处理出错: {str(e)}")
                    continue
            
            avg_loss = epoch_loss / max(n_batches, 1)  # 避免除零
            logger.info(f"轮次 {epoch+1}/{max_epochs}, 损失: {avg_loss:.4f}")
            
            # 验证阶段
            if val_dl is not None:
                val_loss = 0.0
                n_val_batches = 0
                
                for batch in val_dl:
                    try:
                        output = self.training_plan.validate_step(batch)
                        val_loss += output["validation_loss"]
                        n_val_batches += 1
                    except Exception as e:
                        logger.error(f"验证批次处理出错: {str(e)}")
                        continue
                
                avg_val_loss = val_loss / max(n_val_batches, 1)  # 避免除零
                logger.info(f"验证损失: {avg_val_loss:.4f}")
        
        # 训练完成后设置状态
        self.is_trained_ = True
        if hasattr(self.module, "eval"):
            self.module.eval()
        else:
            self.training_plan._set_module_training(False)
        
        return self
    
    def to_device(self, device):
        """将模型移至特定设备。
        
        MLX自动处理设备放置，因此这是一个空操作。
        
        参数
        ----------
        device
            目标设备。
        """
        logger.info("MLX自动处理设备放置，忽略to_device调用")
        pass
    
    @property
    def device(self):
        """获取当前设备。
        
        MLX自动处理设备放置。
        
        返回
        -------
        str
            设备标识符。
        """
        return "mlx"