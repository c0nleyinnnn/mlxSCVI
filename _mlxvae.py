# src/scvi/module/_mlxvae.py

import mlx.core as mx
import mlx.nn as nn
from typing import Dict, Tuple, Optional, List, Any, Union

from scvi import REGISTRY_KEYS
from scvi.module.base import BaseModuleClass, LossOutput

class MlxDense(nn.Module):
    """MLX密集层，使用与PyTorch兼容的初始化方式。"""
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        scale = 1 / 3
        limit = scale * (6 / (in_features + out_features)) ** 0.5
        self.weight = mx.random.uniform(low=-limit, high=limit, shape=(in_features, out_features))
        self.bias = mx.zeros(out_features)
    
    def __call__(self, x):
        return x @ self.weight + self.bias

class MlxEncoder(nn.Module):
    """MLX VAE的编码器模块。"""
    
    def __init__(self, n_input: int, n_latent: int, n_hidden: int, dropout_rate: float):
        super().__init__()
        self.dense1 = MlxDense(n_input, n_hidden)
        self.bn1 = nn.BatchNorm(n_hidden, momentum=0.9)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dense2 = MlxDense(n_hidden, n_hidden)
        self.bn2 = nn.BatchNorm(n_hidden, momentum=0.9)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.mean_layer = MlxDense(n_hidden, n_latent)
        self.var_layer = MlxDense(n_hidden, n_latent)
    
    def __call__(self, x):
        x = mx.log1p(x)
        h = self.dense1(x)
        h = self.bn1(h)
        h = nn.relu(h)
        h = self.dropout1(h)
        h = self.dense2(h)
        h = self.bn2(h)
        h = nn.relu(h)
        h = self.dropout2(h)
        mean = self.mean_layer(h)
        log_var = self.var_layer(h)
        var = mx.exp(log_var)
        return mean, var

class MlxDecoder(nn.Module):
    """MLX VAE的解码器模块。"""
    
    def __init__(self, n_input: int, n_hidden: int, n_batch: int, n_latent: int, dropout_rate: float = 0.0):
        super().__init__()
        self.dense1 = MlxDense(n_latent, n_hidden)
        self.dense2 = MlxDense(n_batch, n_hidden)
        self.bn1 = nn.BatchNorm(n_hidden, momentum=0.9)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dense3 = MlxDense(n_hidden, n_hidden)
        self.dense4 = MlxDense(n_batch, n_hidden)
        self.bn2 = nn.BatchNorm(n_hidden, momentum=0.9)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dense5 = MlxDense(n_hidden, n_input)
        key = mx.random.key(0)
        self.disp = mx.random.normal(key=key, shape=(n_input,))
    
    def __call__(self, z, batch):
        h = self.dense1(z) + self.dense2(batch)
        h = self.bn1(h)
        h = nn.relu(h)
        h = self.dropout1(h)
        h = self.dense3(h) + self.dense4(batch)
        h = self.bn2(h)
        h = nn.relu(h)
        h = self.dropout2(h)
        rho_unnorm = self.dense5(h)
        return rho_unnorm, self.disp

class MlxVAE(nn.Module):
    """使用MLX框架的变分自编码器模型。"""
    
    def __init__(
        self,
        n_input: int,
        n_batch: int,
        n_hidden: int = 128,
        n_latent: int = 30,
        dropout_rate: float = 0.0,
        n_layers: int = 1,
        gene_likelihood: str = "nb",
        eps: float = 1e-8,
    ):
        super().__init__()
        self.n_input = n_input
        self.n_batch = n_batch
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.dropout_rate = dropout_rate
        self.n_layers = n_layers
        self.gene_likelihood = gene_likelihood
        self.eps = eps
        self.encoder = MlxEncoder(n_input, n_latent, n_hidden, dropout_rate)
        self.decoder = MlxDecoder(n_input, n_hidden, n_batch, n_latent, dropout_rate)
    
    def _get_inference_input(self, tensors: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """获取推断模型的输入。"""
        x = tensors[REGISTRY_KEYS.X_KEY]
        return {"x": x}
    
    def _get_generative_input(
        self, 
        tensors: Dict[str, mx.array],
        inference_outputs: Dict[str, mx.array],
    ) -> Dict[str, mx.array]:
        """获取生成模型的输入。"""
        z = inference_outputs["z"]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        return {
            "z": z,
            "batch_index": batch_index,
        }
    
    def train(self, mode=True):
        """递归设置子模块为训练模式。"""
        super().train(mode)
        for module in self.children():
            if isinstance(module, nn.Module):
                module.train(mode)
        return self
    
    def eval(self):
        """递归设置子模块为评估模式。"""
        super().eval()
        for module in self.children():
            if isinstance(module, nn.Module):
                module.eval()
        return self
    
    def inference(self, x: mx.array, n_samples: int = 1) -> Dict[str, Any]:
        mean, var = self.encoder(x)
        stddev = mx.sqrt(var) + self.eps
        key = mx.random.key(0)
        eps = mx.random.normal(key=key, shape=mean.shape)
        z = mean + stddev * eps
        return {"mean": mean, "var": var, "z": z}
    
    def generative(self, z, batch_index) -> Dict[str, Any]:
        batch = mx.zeros((batch_index.shape[0], self.n_batch))
        rows = mx.arange(batch_index.reshape(-1).shape[0])
        batch = batch.at[rows, batch_index.reshape(-1)].add(1.0)
        rho_unnorm, disp = self.decoder(z, batch)
        rho_exp = mx.exp(rho_unnorm)
        rho = rho_exp / mx.sum(rho_exp, axis=-1, keepdims=True)
        return {"rho": rho, "disp": disp}
    
    def loss(self, tensors, inference_outputs, generative_outputs, kl_weight: float = 1.0) -> LossOutput:
        x = tensors[REGISTRY_KEYS.X_KEY]
        disp = generative_outputs["disp"]
        mean = inference_outputs["mean"]
        var = inference_outputs["var"]
        rho = generative_outputs["rho"]
        total_count = mx.sum(x, axis=-1, keepdims=True)
        mu = total_count * rho
        
        eps = 1e-10
        log_theta_mu_eps = mx.log(disp + mu + eps)
        log_theta_eps = mx.log(disp + eps)
        log_mu_eps = mx.log(mu + eps)
        log_prob = x * (log_mu_eps - log_theta_mu_eps) + disp * (log_theta_eps - log_theta_mu_eps)
        log_prob += mx.log1p(mx.exp(log_theta_mu_eps)) - mx.log1p(mx.exp(log_theta_eps))
        reconst_loss = -mx.sum(log_prob, axis=-1)
        kl_divergence = 0.5 * mx.sum(var + mx.square(mean) - 1.0 - mx.log(var + eps), axis=-1)
        weighted_kl = kl_weight * kl_divergence
        loss = mx.mean(reconst_loss + weighted_kl)
        return LossOutput(loss=loss, reconstruction_loss=reconst_loss, kl_local=kl_divergence)
    
    def __call__(self, tensors: Dict[str, mx.array], kl_weight: float = 1.0) -> Tuple[Dict, Dict, LossOutput]:
        inference_inputs = self._get_inference_input(tensors)
        inference_outputs = self.inference(**inference_inputs)
        generative_inputs = self._get_generative_input(tensors, inference_outputs)
        generative_outputs = self.generative(**generative_inputs)
        loss_output = self.loss(tensors, inference_outputs, generative_outputs, kl_weight)
        return inference_outputs, generative_outputs, loss_output