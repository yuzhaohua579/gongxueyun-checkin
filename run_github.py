#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GitHub Actions 专用打卡入口
支持环境变量配置，输出详细日志
"""
import sys
import os

# 设置工作目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
sys.path.insert(0, BASE_DIR)

import logging
import json
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("gongxueyun_github")

# 强制设置时区为北京时间
os.environ['TZ'] = 'Asia/Shanghai'
try:
    import time
    time.tzset()
except Exception:
    pass  # Windows 不支持 tzset，忽略


def run():
    print("=" * 60)
    print("  工学云自动打卡 - GitHub Actions 版")
    print(f"  执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 从环境变量读取配置并生成 config.json
    config_file = os.path.join(BASE_DIR, "user", "config.json")
    if not os.path.exists(config_file):
        logger.error("config.json 不存在！请检查 GitHub Secrets 配置")
        return False

    # 验证配置
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        phone = config_data.get('config', {}).get('user', {}).get('phone', '')
        if not phone or phone.startswith('${{'):
            logger.error("手机号未正确配置！请检查 Secrets 设置")
            return False
        logger.info(f"配置加载成功，手机号: {phone[:3]}****{phone[-4:]}")
    except Exception as e:
        logger.error(f"配置文件读取失败: {e}")
        return False

    # 导入打卡模块
    try:
        from step.login import login
        from step.fetchPlan import fetch_plan
        from step.clockIn import clock_in
        from util.HelperFunctions import get_checkin_types, is_workday_realtime
    except Exception as e:
        logger.error(f"模块导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 1. 检查工作日
    print("\n📅 [1/5] 检查工作日...")
    if not is_workday_realtime():
        logger.info("今天是非工作日，跳过打卡")
        print("  ℹ️  今天不需要打卡（周末/节假日）")
        print("=" * 60)
        return True

    print("  ✅ 今天是工作日")

    # 2. 登录
    print("\n🔐 [2/5] 登录工学云...")
    if not login():
        logger.error("登录失败")
        print("  ❌ 登录失败")
        print("=" * 60)
        return False
    print("  ✅ 登录成功")

    # 3. 获取实习计划
    print("\n📋 [3/5] 获取实习计划...")
    if not fetch_plan():
        logger.error("获取实习计划失败")
        print("  ❌ 获取实习计划失败")
        print("=" * 60)
        return False
    print("  ✅ 获取成功")

    # 4. 判断打卡类型
    print("\n⏰ [4/5] 判断打卡类型...")
    checkin_types = get_checkin_types()
    if not checkin_types:
        logger.info("当前不需要打卡")
        print("  ℹ️  当前时段不需要打卡")
        print("=" * 60)
        return True

    type_names = [t["display"] for t in checkin_types]
    print(f"  需要打卡: {', '.join(type_names)}")

    # 5. 执行打卡
    print("\n✅ [5/5] 执行打卡...")
    success_count = 0
    fail_count = 0
    results = []

    for ctype in checkin_types:
        display = ctype["display"]
        print(f"\n  正在执行【{display}】打卡...")
        result = clock_in(ctype["type"])
        if result:
            print(f"    ✅ {display}打卡成功！")
            success_count += 1
            results.append(f"{display}: 成功")
        else:
            print(f"    ❌ {display}打卡失败")
            fail_count += 1
            results.append(f"{display}: 失败")

    # 输出总结
    print(f"\n📊 打卡完成：成功 {success_count} 次，失败 {fail_count} 次")
    print("=" * 60)

    # 设置 GitHub Actions 输出
    if 'GITHUB_OUTPUT' in os.environ:
        with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
            f.write(f"success_count={success_count}\n")
            f.write(f"fail_count={fail_count}\n")
            f.write(f"results={', '.join(results)}\n")

    return fail_count == 0


if __name__ == "__main__":
    try:
        success = run()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️ 用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 程序异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
