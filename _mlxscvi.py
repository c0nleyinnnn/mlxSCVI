# src/scvi/model/_mlxscvi.py

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, Optional, Sequence

import mlx.core as mx
import numpy as np

from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import CategoricalObsField, LayerField
from scvi.module import MlxVAE  # 继续使用大写M的MlxVAE
from scvi.utils import setup_anndata_dsp

# 导入MlxTrainingMixin（保持大写M）
from .base import BaseModelClass, MlxTrainingMixin

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Literal
    
    import numpy as np
    from anndata import AnnData

logger = logging.getLogger(__name__)

class mlxSCVI(MlxTrainingMixin, BaseModelClass):
    """使用MLX框架的单细胞变分推断模型。
    
    此实现利用了MLX框架的特性，在Apple Silicon芯片上提供优化的性能。
    
    Parameters
    ----------
    adata
        通过mlxSCVI.setup_anndata()注册的AnnData对象。
    n_hidden
        每个隐藏层的节点数量。
    n_latent
        潜在空间的维度。
    dropout_rate
        神经网络的dropout率。
    gene_likelihood
        以下之一：
        * 'nb' - 负二项分布
        * 'poisson' - 泊松分布
    """
    
    _module_cls = MlxVAE
    
    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 128,
        n_latent: int = 10,
        dropout_rate: float = 0.1,
        gene_likelihood: Literal["nb", "poisson"] = "nb",
        **model_kwargs,
    ):
        super().__init__(adata)
        
        n_batch = self.summary_stats.n_batch
        
        self.module = self._module_cls(
            n_input=self.summary_stats.n_vars,
            n_batch=n_batch,
            n_hidden=n_hidden,
            n_latent=n_latent,
            dropout_rate=dropout_rate,
            gene_likelihood=gene_likelihood,
            **model_kwargs,
        )
        
        self._model_summary_string = ""
        self.init_params_ = self._get_init_params(locals())
    
    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: str | None = None,
        batch_key: str | None = None,
        **kwargs,
    ):
        """设置AnnData对象用于训练。
        
        Parameters
        ----------
        adata
            AnnData对象。
        layer
            如果不是None，使用该层而不是X进行训练。
        batch_key
            如果不是None，使用该键指定的obs列作为批次信息。
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
        ]
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)
    
    def get_latent_representation(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        give_mean: bool = True,
        n_samples: int = 1,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """获取每个细胞的潜在表示。
        
        Parameters
        ----------
        adata
            与初始AnnData对象结构相同的AnnData对象。如果为None，默认使用初始化模型时的AnnData对象。
        indices
            adata中要使用的细胞索引。如果为None，使用所有细胞。
        give_mean
            是否返回后验分布的均值或样本。
        n_samples
            用于计算潜在表示的样本数量。
        batch_size
            用于数据加载到模型的小批量大小。
        
        Returns
        -------
        latent_representation : np.ndarray
            每个细胞的低维表示
        """
        self._check_if_trained(warn=False)
        
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size, iter_ndarray=True
        )
        
        # 设置为评估模式
        self.module.eval()
        
        latent = []
        for array_dict in scdl:
            # 转换为MLX数组
            mlx_dict = {k: mx.array(v) for k, v in array_dict.items()}
            outputs = self.module.inference(mlx_dict[REGISTRY_KEYS.X_KEY], n_samples=n_samples)
            
            if give_mean:
                z = outputs["mean"]
            else:
                z = outputs["z"]
            
            # 使用标准方式转换为NumPy数组
            latent.append(np.array(z.tolist()))
        
        # 连接所有批次
        latent = np.concatenate(latent, axis=0)
        
        return latent
    
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