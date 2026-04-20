import torch
import numpy as np
import time
from fastapi import HTTPException
from app.ml.GINEClassifier import CostModelV2
from app.ml import Normalize as nz
from app.models.predict_model import PredictRequest, PredictResponse
from app.config import GlobalConfig as config
from concurrent.futures import ThreadPoolExecutor
import asyncio
import threading

executor = ThreadPoolExecutor(max_workers=11)


class PredictService:
    _instance = None
    _init_lock = threading.Lock()  # 单例初始化锁，防止多线程同时创建实例

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._init_lock:  # 双重检查锁，确保单例线程安全
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model  = self._load_model()
        self._lock  = None          # 异步锁懒加载，事件循环启动后才创建
        self._infer_lock = threading.Lock()  # 推理线程锁，确保模型推理线程安全
        print(f"服务初始化完成，使用设备: {self.device}")

    def _load_model(self):
        model = CostModelV2()
        model.load_state_dict(torch.load('best_model.pt', map_location=self.device))
        model.to(self.device)
        model.eval()
        print(f"模型加载完成，使用设备: {self.device}")
        return model

    @property
    def lock(self):
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    # json调用
    async def predict(self, req: PredictRequest) -> PredictResponse:
        start = time.time()

        # 参数校验
        self._validate(req)

        # JSON转numpy
        x          = np.array(req.x,          dtype=np.float32)
        edge_index = np.array(req.edge_index, dtype=np.int32)
        edge_attr  = np.array(req.edge_attr,  dtype=np.float32)

        # 标准化,java里做了标准化这里就不用了
        # edge_attr, x = nz.normalize_all(edge_attr, x)

        # 转tensor并移到设备
        x_t          = torch.tensor(x,          dtype=torch.float).to(self.device)
        edge_index_t = torch.tensor(edge_index, dtype=torch.long).to(self.device)
        edge_attr_t  = torch.tensor(edge_attr,  dtype=torch.float).to(self.device)

        # 推理放到线程池，不阻塞事件循环
        loop = asyncio.get_event_loop()
        pred = await loop.run_in_executor(
            executor, self._do_infer, x_t, edge_index_t, edge_attr_t
        )

        elapsed = (time.time() - start) * 1000
        print(f"预测完成，结果: {pred:.4f}，耗时: {elapsed:.1f}ms")

        return PredictResponse(
            predicted_cost=pred,
            elapsed_ms=round(elapsed, 1)
        )

    # 模型预测
    def _do_infer(self, x_t, edge_index_t, edge_attr_t) -> float:
        # 线程锁保护模型推理，确保同一时刻只有一个线程在推理
        # with self._infer_lock:
        with torch.no_grad():
            pred = self.model(x_t, edge_index_t, edge_attr_t)
        return pred.item()

    # 参数验证
    def _validate(self, req: PredictRequest):
        if len(req.x) != config.NUM_NODES or len(req.x[0]) != config.NODE_FEAT_DIM:
            raise HTTPException(
                status_code=400,
                detail=f"x维度错误，期望[{config.NUM_NODES},{config.NODE_FEAT_DIM}]，"
                       f"实际[{len(req.x)},{len(req.x[0])}]"
            )
        if len(req.edge_index) != 2 or len(req.edge_index[0]) != config.NUM_BRANCHES:
            raise HTTPException(
                status_code=400,
                detail=f"edge_index维度错误，期望[2,{config.NUM_BRANCHES}]，"
                       f"实际[{len(req.edge_index)},{len(req.edge_index[0])}]"
            )
        if len(req.edge_attr) != config.NUM_BRANCHES or len(req.edge_attr[0]) != config.EDGE_FEAT_DIM:
            raise HTTPException(
                status_code=400,
                detail=f"edge_attr维度错误，期望[{config.NUM_BRANCHES},{config.EDGE_FEAT_DIM}]，"
                       f"实际[{len(req.edge_attr)},{len(req.edge_attr[0])}]"
            )

    # 二进制预测
    async def predict_binary(self, body: bytes) -> PredictResponse:
        start = time.time()

        offset = 0
        edge_index = np.frombuffer(body[offset:offset+1688],  dtype='>i4').reshape(2, 211).byteswap().newbyteorder('=').copy()
        offset += 1688
        edge_attr  = np.frombuffer(body[offset:offset+3376],  dtype='>f4').reshape(211, 4).byteswap().newbyteorder('=').copy()
        offset += 3376
        x          = np.frombuffer(body[offset:offset+123200], dtype='>f4').reshape(175, 176).byteswap().newbyteorder('=').copy()


        t1 = time.time()

        # 标准化
        # edge_attr, x = nz.normalize_all(edge_attr, x)

        loop = asyncio.get_event_loop()
        pred = await loop.run_in_executor(
            executor, self._prepare_and_infer, x, edge_index, edge_attr
        )

        elapsed = (time.time() - start) * 1000
        t_parse = (t1 - start) * 1000
        print(f"解析耗时: {t_parse:.1f}ms，总耗时: {elapsed:.1f}ms")

        return PredictResponse(predicted_cost=pred, elapsed_ms=round(elapsed, 1))
    def _prepare_and_infer(self, x, edge_index, edge_attr) -> float:
        x_t          = torch.tensor(x,          dtype=torch.float).to(self.device)
        edge_index_t = torch.tensor(edge_index, dtype=torch.long).to(self.device)
        edge_attr_t  = torch.tensor(edge_attr,  dtype=torch.float).to(self.device)
        return self._do_infer(x_t, edge_index_t, edge_attr_t)