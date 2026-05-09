from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from pydantic import BaseModel


class RewardFuncInfo(BaseModel):
    name: str  # 奖励函数名称
    reward_fn: Callable[[List], List]  # 奖励函数对象
    labels: List[str] = field(default_factory=list)  # 奖励函数作用的label列表
    integration: str = "sum"  # 多奖励合并规则 sum, multiply
    reward_fn_key: str = "data_source"
