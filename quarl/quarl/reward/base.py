import importlib.util
import sys
from functools import partial
from typing import Dict

from quarl.interface import RewardFuncInfo

from .score.bad_pattern import compute_score_batch as reward_bad_pattern_fn

REWARD_FUNCTIONS = {
    "bad_pattern": reward_bad_pattern_fn,
}


def get_custom_reward_fns(config) -> Dict[str, RewardFuncInfo]:
    """
    根据配置文件导入多个奖励函数

    Args:
        config: 配置对象，包含了奖励函数的配置信息

    Returns:
        以reward function 名为key的字典，value为RewardFuncInfo类
    """

    # 获取奖励函数的配置
    reward_fns_config = config.get("custom_reward_functions", {})
    if not reward_fns_config:
        return {}

    reward_func_dict = {}
    print(f"using customized reward functions: {reward_fns_config}")
    # 处理每个配置的奖励函数
    for reward_fn_name, reward_config in reward_fns_config.items():
        if reward_fn_name not in REWARD_FUNCTIONS:
            print(f"[quarl.reward]: Warning: reward_fn `{reward_fn_name}` not exists!")
            continue

        reward_fn = REWARD_FUNCTIONS[reward_fn_name]
        kwargs = reward_config.get("kwargs", {})
        # 添加到结果列表

        reward_func_dict[reward_fn_name] = RewardFuncInfo(
            name=reward_fn_name,
            reward_fn=partial(reward_fn, **kwargs),
            labels=reward_config.get("labels", []),
            integration=reward_config.get("integration", "sum"),
        )

        print(f"Successfully load reward_fn: {reward_fn_name}")

    return reward_func_dict
