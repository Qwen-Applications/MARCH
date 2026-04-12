from dataclasses import field
from typing import Dict, List, Optional

from pydantic import BaseModel
from verl.workers.rollout.schemas import Message


class BaseDataItem(BaseModel):
    prompt: List[Dict]  # 数据的prompt，以messages格式给出
    uid: Optional[str] = None  # 数据的uid
    data_source: Optional[str] = None  # 数据集标签
    tags: List[str] = field(default_factory=list)  # 数据的类别标签tag集合，不同tag用于套用不同的训练策略


class RMDataItem(BaseDataItem):
    chosen: str  # RM偏好中的chosen response
    rejected: str = ""  # RM偏好中的rejected response
    chosen_score: Optional[float] = None
    rejected_score: Optional[float] = None


class RewardInfo(BaseModel):
    style: str
    groud_truth: str


class RLDataItem(BaseDataItem):
    ground_truth: Optional[str] = None  # 标准答案
    reward_model: RewardInfo
    ability: Optional[str] = None
    extra_info: Optional[Dict] = field(default_factory=dict)
