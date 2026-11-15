import logging
import threading
from datetime import datetime, timedelta

import requests

from manager.ConfigManager import ConfigManager

# 尝试导入主模块的日志上下文，失败则创建本地版本
try:
    from main import _log_ctx
except ImportError:
    _log_ctx = threading.local()

logger = logging.getLogger(__name__)


def get_current_month_info() -> dict:
    """
    获取当前月份的开始和结束时间。

    该方法计算当前月份的开始日期和结束日期，并将它们返回为字典，
    字典中包含这两项的字符串表示。

    Returns:
        包含当前月份开始和结束时间的字典。
    """
    now = datetime.now()
    # 当前月份的第一天
    start_of_month = datetime(now.year, now.month, 1)

    # 下个月的第一天
    if now.month == 12:
        next_month_start = datetime(now.year + 1, 1, 1)
    else:
        next_month_start = datetime(now.year, now.month + 1, 1)

    # 当前月份的最后一天（下个月第一天减一天）
    end_of_month = next_month_start - timedelta(days=1)

    # 格式化为字符串
    start_time_str = start_of_month.strftime("%Y-%m-%d %H:%M:%S")
    end_time_str = end_of_month.strftime("%Y-%m-%d 00:00:00Z")

    return {"startTime": start_time_str, "endTime": end_time_str}


def desensitize_name(name: str) -> str:
    """
    对姓名进行脱敏处理，将中间部分字符替换为星号。

    Args:
        name (str): 待脱敏的姓名。

    Returns:
        str: 脱敏后的姓名。
    """
    name = name.strip()  # 去除前后空格，防止输入有空格影响判断

    n = len(name)
    if n < 3:
        return f"{name[0]}*"
    else:
        return f"{name[0]}{'*' * (n - 2)}{name[-1]}"


def is_workday_realtime() -> bool:
    """
    实时判断今天是否需要打卡（考虑法定节假日 + 自定义工作日）。

    逻辑：
    1. 如果 holidaysClockIn=false 且今天是法定节假日（非调休）→ 不打卡
    2. 检查今天是否在 customDays 范围内（1=周一, 7=周日）→ 不在则不打卡
    3. 其余情况 → 打卡

    返回值:
        bool: True 表示今天需要打卡，False 表示不需要
    """

    check_date = datetime.today()
    date_str = check_date.strftime("%Y-%m-%d")

    # 1. 检查自定义工作日
    custom_days = ConfigManager.get("clockIn", "customDays", default=[1, 2, 3, 4, 5])
    today_weekday_num = check_date.weekday() + 1  # Python weekday: 0=Mon → 1=Mon
    if today_weekday_num not in custom_days:
        logging.info(f"{date_str} 是周{today_weekday_num}，不在自定义打卡日 {custom_days} 中，跳过")
        return False

    # 2. 检查是否为法定节假日（如果 holidaysClockIn=false）
    holidays_clock_in = ConfigManager.get("clockIn", "holidaysClockIn", default=False)
    if not holidays_clock_in:
        url = f"https://timor.tech/api/holiday/info/{date_str}"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        }
        try:
            resp = requests.get(url, headers=headers, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("code") == 0:
                    # type.type: 0-工作日 1-周末 2-节假日 3-调休日(工作日)
                    day_type = data.get("type", {}).get("type")
                    if day_type == 2:  # 法定节假日
                        logging.info(f"{date_str} 是法定节假日，跳过打卡")
                        return False
                    logging.info(f"{date_str} 日期类型: {day_type}，需要打卡")
        except Exception as e:
            logging.warning(f"节假日API调用异常: {e}，按自定义工作日处理")

    logging.info(f"{date_str} 是工作日，需要打卡")
    return True


def get_checkin_type() -> dict[str, str]:
    """
    获取打卡类型（单次模式）。

    该方法根据配置文件获取打卡类型，并返回一个字典，包含打卡类型和显示名称。
    适用于 single 模式（只打一次卡）。

    Returns:
        dict[str, str]: 包含打卡类型和显示名称的字典。
    """
    type = ConfigManager.get("clockIn", "type")
    if type == "START":
        return {"type": "START", "display": "上班"}
    elif type == "END":
        return {"type": "END", "display": "下班"}
    else:
        return {"type": "HOLIDAY", "display": "休息/节假日"}


def get_checkin_types() -> list[dict[str, str]]:
    """
    根据打卡模式获取打卡类型列表。

    支持的模式：
    - twice_daily: 一天两次打卡（上班 + 下班）
    - single: 单次打卡，根据 clockIn.type 决定类型
    - 其他/未配置: 默认休息/节假日打卡

    Returns:
        list[dict[str, str]]: 打卡类型列表，每个元素包含 type 和 display。
    """
    mode = ConfigManager.get("clockIn", "mode", default="single")
    if mode == "twice_daily":
        return [
            {"type": "START", "display": "上班"},
            {"type": "END", "display": "下班"},
        ]
    else:
        return [get_checkin_type()]
