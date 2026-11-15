import logging

from util.ApiService import ApiService

logger = logging.getLogger(__name__)


def clock_in(checkin_type: str = "START") -> bool:
    """
    执行打卡操作

    Args:
        checkin_type (str): 打卡类型，"START" 上班，"END" 下班

    Returns:
        bool: 打卡成功返回True，失败返回False
    """
    type_display = "上班" if checkin_type == "START" else "下班"
    logging.info(f"开始执行{type_display}打卡")

    try:
        api_client = ApiService()
        checkin_info = {
            "type": checkin_type,
            "description": "",
        }
        result = api_client.submit_clock_in(checkin_info)

        if result and result.get("result"):
            rsp_data = result.get("data", {})
            code = rsp_data.get("code")
            msg = rsp_data.get("msg", "")

            if code == 200:
                logger.info(f"{type_display}打卡成功")
                return True
            else:
                logger.warning(f"{type_display}打卡失败: code={code}, msg={msg}")
                return False
        else:
            logger.warning(f"{type_display}打卡失败: 返回结果异常")
            return False

    except Exception as e:
        logger.exception(f"{type_display}打卡过程发生异常: {e}")
        return False
