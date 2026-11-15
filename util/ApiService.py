import json
import logging
import random
import re
import time
import uuid
from typing import Dict, Any, List, Optional

import requests

from manager.ConfigManager import ConfigManager
from manager.PlanInfoManager import PlanInfoManager
from manager.UserInfoManager import UserInfoManager
from util.CaptchaUtils import recognize_blockPuzzle_captcha, recognize_clickWord_captcha
from util.CryptoUtils import create_sign, aes_encrypt, aes_decrypt
from util.HelperFunctions import get_current_month_info

logger = logging.getLogger(__name__)

# 常量
BASE_URL = "https://api.moguding.net:9000/"
HEADERS = {
    "user-agent": "Dart/2.17 (dart:io)",
    "content-type": "application/json; charset=utf-8",
    "accept-encoding": "gzip",
    "host": "api.moguding.net:9000",
}


class ApiService:
    def __init__(self):

        self.max_retries = 5  # 控制重新尝试的次数

    def _post_request(
            self,
            url: str,
            headers: Dict[str, str],
            data: Dict[str, Any],
            retry_count: int = 0,
    ) -> Dict[str, Any]:
        """
        发送POST请求，并处理请求过程中可能发生的错误。
        包括自动重试机制和Token失效处理。
        """
        try:
            response = requests.post(f"{BASE_URL}{url}",
                                     headers=headers,
                                     json=data,
                                     timeout=10)
            response.raise_for_status()
            rsp = response.json()

            if rsp.get("code") == 200 and rsp.get("msg", "未知错误") == "302":
                return rsp

            if rsp.get("code") == 200 or rsp.get("code") == 6111:
                return rsp

            if ("token失效" in rsp.get("msg", "未知错误")
                    and retry_count < self.max_retries):
                wait_time = 1 * (2 ** retry_count)
                time.sleep(wait_time)
                logger.warning("Token失效，正在重新登录...")
                if self.login():
                    new_token = UserInfoManager.get_token()
                    headers["authorization"] = new_token
                    logger.info("已更新 Authorization Token，重试请求")
                    return self._post_request(url, headers, data, retry_count + 1)
            else:
                logger.info(f"服务端返回详情: code={rsp.get('code')}, msg={rsp.get('msg')}, data={rsp.get('data')}")
                raise ValueError(rsp.get("msg", "未知错误"))

        except (requests.RequestException, ValueError) as e:
            if re.search(r"[\u4e00-\u9fff]",
                         str(e)) or retry_count >= self.max_retries:
                raise ValueError(f"{str(e)}")

            wait_time = 1 * (2 ** retry_count)
            logger.warning(
                f"重试 {retry_count + 1}/{self.max_retries}，等待 {wait_time:.2f} 秒"
            )
            time.sleep(wait_time)

        return self._post_request(url, headers, data, retry_count + 1)

    def pass_blockPuzzle_captcha(self, max_attempts: int = 5) -> str:
        """通过行为验证码（滑块验证）。"""
        attempts = 0
        while attempts < max_attempts:
            captcha_url = "session/captcha/v1/get"
            request_data = {
                "clientUid": str(uuid.uuid4()).replace("-", ""),
                "captchaType": "blockPuzzle",
            }
            captcha_info = self._post_request(
                captcha_url,
                HEADERS,
                request_data,
            )
            slider_data = recognize_blockPuzzle_captcha(
                captcha_info["data"]["jigsawImageBase64"],
                captcha_info["data"]["originalImageBase64"],
            )
            check_slider_url = "session/captcha/v1/check"
            check_slider_data = {
                "pointJson":
                    aes_encrypt(slider_data, captcha_info["data"]["secretKey"],
                                "b64"),
                "token":
                    captcha_info["data"]["token"],
                "captchaType":
                    "blockPuzzle",
            }
            check_result = self._post_request(
                check_slider_url,
                HEADERS,
                check_slider_data,
            )
            if check_result.get("code") != 6111:
                return aes_encrypt(
                    captcha_info["data"]["token"] + "---" + slider_data,
                    captcha_info["data"]["secretKey"],
                    "b64",
                )
            attempts += 1
            time.sleep(random.uniform(1, 3))
        raise Exception("通过滑块验证码失败")

    def solve_click_word_captcha(self, max_retries: int = 2) -> str:
        """通过文字点选验证码。"""
        retry_count = 0
        while retry_count < max_retries:
            captcha_endpoint = "/attendence/clock/v1/get"
            captcha_request_payload = {
                "clientUid": str(uuid.uuid4()).replace("-", ""),
                "captchaType": "clickWord",
            }
            captcha_response = self._post_request(
                captcha_endpoint,
                self._get_authenticated_headers(),
                captcha_request_payload,
            )
            captcha_solution = recognize_clickWord_captcha(
                captcha_response["data"]["originalImageBase64"],
                captcha_response["data"]["wordList"],
            )
            verification_endpoint = "/attendence/clock/v1/check"
            verification_payload = {
                "pointJson":
                    aes_encrypt(captcha_solution,
                                captcha_response["data"]["secretKey"],
                                "b64"),
                "token":
                    captcha_response["data"]["token"],
                "captchaType":
                    "clickWord",
            }
            try:
                verification_response = self._post_request(
                    verification_endpoint,
                    self._get_authenticated_headers(),
                    verification_payload,
                )
            except ValueError:
                logger.warning("验证码校验请求失败，重试中...")
                retry_count += 1
                time.sleep(random.uniform(1, 3))
                continue

            if verification_response.get("code") == 200:
                encrypted_result = aes_encrypt(
                    captcha_response["data"]["token"] + "---" +
                    captcha_solution,
                    captcha_response["data"]["secretKey"],
                    "b64",
                )
                return encrypted_result

            retry_count += 1
            time.sleep(random.uniform(1, 3))

        raise Exception("通过点选验证码失败")

    def _get_authenticated_headers(
            self,
            sign_data: Optional[List[Optional[str]]] = None
    ) -> Dict[str, str]:
        """生成带有认证信息的请求头。"""
        headers = {
            **HEADERS,
            "authorization": UserInfoManager.get_token(),
            "userid": UserInfoManager.get_userid(),
            "rolekey": UserInfoManager.get("roleKey"),
            "version": "5.31.6",
        }
        if sign_data:
            headers["sign"] = create_sign(*sign_data)
        return headers

    def login(self) -> bool:
        """执行用户登录操作。"""
        logger.info("执行登录")

        try:
            url = "session/user/v6/login"
            data = {
                "phone": aes_encrypt(ConfigManager.get("user", "phone")),
                "password": aes_encrypt(ConfigManager.get("user", "password")),
                "captcha": self.pass_blockPuzzle_captcha(),
                "loginType": "android",
                "uuid": str(uuid.uuid4()).replace("-", ""),
                "device": "android",
                "version": "5.31.6",
                "t": aes_encrypt(str(int(time.time() * 1000))),
            }

            response = self._post_request(url, HEADERS, data)

            encrypted_data = response.get("data")
            if not encrypted_data:
                logger.error("登录失败：返回数据为空")
                return False

            user_info = json.loads(aes_decrypt(encrypted_data))
            logger.info(f"登录结果：{user_info}")

            UserInfoManager.set_userinfo(user_info)
            logger.info("用户信息已保存到 UserInfoManager 管理的文件和缓存中")
            return True

        except Exception as e:
            logger.exception(f"登录过程发生异常：{e}")
            return False

    def fetch_plan(self) -> bool:
        """获取当前用户的实习计划并更新 PlanInfoManager。"""
        try:
            url = "practice/plan/v3/getPlanByStu"
            data = {
                "pageSize": 999999,
                "t": aes_encrypt(str(int(time.time() * 1000)))
            }
            headers = self._get_authenticated_headers(sign_data=[
                UserInfoManager.get_userid(),
                UserInfoManager.get("roleKey"),
            ])

            rsp = self._post_request(url, headers, data)

            data_list = rsp.get("data")
            if not data_list or not isinstance(data_list, list):
                logger.warning("未获取到实习计划数据")
                return False

            plan_info = data_list[0]
            if not plan_info:
                logger.warning("实习计划数据为空")
                return False

            PlanInfoManager.set_planinfo(plan_info)
            logger.info("实习计划信息已更新到 PlanInfoManager")
            return True

        except Exception as e:
            logger.exception("获取实习计划过程中发生异常")
            return False

    def submit_clock_in(self, checkin_info: Dict[str, Any]) -> dict[str, dict[str, Any] | bool] | None:
        """提交打卡信息。"""
        url = "attendence/clock/teacher/v2/save"
        sign_data = None
        planId = PlanInfoManager.get_plan_id()

        if UserInfoManager.get("userType") != "teacher":
            url = "attendence/clock/v6/save"
            sign_data = [
                ConfigManager.get("device"),
                checkin_info.get("type"),
                planId,
                UserInfoManager.get_userid(),
                ConfigManager.get("clockIn", "location", "address")
            ]

        logger.info(f'打卡类型：{checkin_info.get("type")}')

        data = {
            "distance": None,
            "content": None,
            "lastAddress": None,
            "lastDetailAddress": checkin_info.get("lastDetailAddress"),
            "attendanceId": None,
            "country": "中国",
            "createBy": None,
            "createTime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "description": checkin_info.get("description", None),
            "device": ConfigManager.get("device"),
            "images": None,
            "isDeleted": None,
            "isReplace": None,
            "modifiedBy": None,
            "modifiedTime": None,
            "schoolId": None,
            "state": "NORMAL",
            "teacherId": None,
            "teacherNumber": None,
            "type": checkin_info.get("type"),
            "stuId": None,
            "planId": planId,
            "attendanceType": None,
            "username": None,
            "attachments": checkin_info.get("attachments", None),
            "userId": UserInfoManager.get_userid(),
            "isSYN": None,
            "studentId": None,
            "applyState": None,
            "studentNumber": None,
            "memberNumber": None,
            "headImg": None,
            "attendenceTime": None,
            "depName": None,
            "majorName": None,
            "className": None,
            "logDtoList": None,
            "isBeyondFence": None,
            "practiceAddress": None,
            "tpJobId": None,
            "t": aes_encrypt(str(int(time.time() * 1000))),
            "version": "5.31.6",
        }

        data.update(ConfigManager.get("clockIn", "location"))

        headers = self._get_authenticated_headers(sign_data)

        responses = self._post_request(url, headers, data)
        if responses.get("msg") == "302":
            logger.info("检测到行为验证码，正在通过···")
            data["captcha"] = self.solve_click_word_captcha()
            rsp = self._post_request(url, headers, data)
            logger.info(f"打卡结果: {rsp}")
            return {"result": True, "data": rsp}
        else:
            return {"result": True, "data": responses}
