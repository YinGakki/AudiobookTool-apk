from flask import Flask, render_template, jsonify, redirect, url_for, request, Response
from functools import wraps
import os
import requests
import logging
from datetime import datetime
import json
from urllib.parse import urlparse, urlunparse # 这个之前已经建议添加了
import time
import pytz
from datetime import timedelta

# --- 配置认证信息 ---
USERNAME = "FY"  # 你可以修改用户名
PASSWORD = "9863286Fy"  # 改成你自己的密码

# --- 在 USERNAME, PASSWORD 附近添加 ---
API_TOKEN = "cnb-tts-api-url-win10" # 请换成你自己的随机字符串


# --- 先创建app实例！这是关键修复 ---
MONITOR_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE_PATH = os.path.join(MONITOR_DIR, "app.log")
API_JSON_PATH = os.path.join(MONITOR_DIR, "api.json")
KEY_LOG_PATH = os.path.join(MONITOR_DIR, "key.log")
CONFIG_JSON_PATH = os.path.join(MONITOR_DIR, "config.json") 
API_POOL_FILE_PATH = os.path.join(MONITOR_DIR, "api.json") # <--- 新增这一行

app = Flask(__name__, template_folder="templates")

# --- 认证装饰器 ---
def check_auth(username, password):
    """验证用户名密码"""
    return username == USERNAME and password == PASSWORD

def authenticate():
    """返回401认证响应（纯英文，避免编码问题）"""
    return Response(
        'Authentication required. Please log in.',
        401,
        {'WWW-Authenticate': 'Basic realm="Log Monitor System"'},
    )

FAIL_STRING = "所有文本块都未能成功处理"
THRESHOLD = 6 
RESET_TIME_HOUR = 16
RESET_TIME_MINUTE = 30
TIMEZONE = 'Asia/Shanghai'
INTERVAL_SECONDS = 600 # 这个在Web服务中不直接使用，但可以保留
TARGET_MODEL = "gemini"
TARGET_SECTION = "models"

SERVER_URL = "http://127.0.0.1:8000"  # 假设你的后端运行在8000端口，请根据实际情况修改
UPDATE_ENDPOINT = "/api/update_llm_config" # 后端用于热加载配置的端点

# ----------------- 辅助函数 -----------------

def get_beijing_time():
    """获取当前的北京时间 (Asia/Shanghai)"""
    return datetime.now(pytz.timezone(TIMEZONE))

def parse_iso_time(iso_string):
    """
    从 ISO 格式字符串解析带时区信息的 datetime 对象，并确保时区正确。
    """
    try:
        # fromisoformat 可以处理带时区的字符串
        dt = datetime.fromisoformat(iso_string)
        # 如果解析出来的对象是 naive 的，则强制将其视为目标时区的时间
        if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
            tz = pytz.timezone(TIMEZONE)
            dt = tz.localize(dt)
        # 确保是北京时间，以便与 last_reset_bjt 比较
        return dt.astimezone(pytz.timezone(TIMEZONE))
    except Exception as e:
        logging.error(f"解析 ISO 时间字符串失败 ({iso_string}): {e}")
        return None

def get_log_fail_count(log_path, num_lines=100):
    """
    读取日志文件最新的N行，并统计失败字符串的出现次数。
    """
    if not os.path.exists(log_path):
        logging.warning(f"日志文件不存在: {log_path}")
        return 0
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            # 读取最新的 N 行
            lines = f.readlines()
            latest_lines = lines[-num_lines:]
            
            count = 0
            for line in latest_lines:
                # 核心判断逻辑，查找最终错误字符串
                if FAIL_STRING in line:
                    count += 1
            
            logging.info(f"最新 {num_lines} 行日志中，失败字符串出现 {count} 次。")
            return count
            
    except Exception as e:
        logging.error(f"读取或解析日志文件时发生错误: {e}")
        return 0

def load_json(filepath):
    """加载 JSON 文件内容，处理文件不存在和解析错误"""
    if not os.path.exists(filepath):
        # 尝试创建空文件以避免后续错误
        with open(filepath, 'w', encoding='utf-8') as f:
            if filepath == API_POOL_FILE_PATH:
                json.dump([], f, ensure_ascii=False, indent=4)
            elif filepath == CONFIG_JSON_PATH:
                # config文件初始化为示例结构
                json.dump({TARGET_SECTION: {TARGET_MODEL: {"api_key": "INITIAL_PLACEHOLDER_KEY"}}}, f, ensure_ascii=False, indent=4)
        logging.warning(f"文件不存在，已创建示例文件: {filepath}")
        # 重新尝试加载（如果创建成功）
        if os.path.exists(filepath):
            return load_json(filepath)
        return None

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logging.error(f"JSON 文件解析错误 ({filepath}): {e}")
        return None
    except Exception as e:
        logging.error(f"加载文件时发生错误 ({filepath}): {e}")
        return None

def save_json(filepath, data):
    """保存 JSON 文件内容"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return True
    except Exception as e:
        logging.error(f"保存文件时发生错误 ({filepath}): {e}")
        return False

def check_and_reset_keys(api_pool_data):
    """
    检查并重置密钥状态。
    只有当密钥的最后使用时间 'last_used' 早于最近一次经过的重置时间点时才重置。
    """
    now_bjt = get_beijing_time()
    tz = pytz.timezone(TIMEZONE)

    # 1. 确定今天的重置时间点 (例如今天的 16:30:00)
    today_reset_bjt = tz.localize(datetime(
        year=now_bjt.year,
        month=now_bjt.month,
        day=now_bjt.day,
        hour=RESET_TIME_HOUR,
        minute=RESET_TIME_MINUTE,
        second=0,
        microsecond=0
    ))
    
    # 2. 确定最近一次经过的重置时间点 (last_reset_bjt)
    # 如果当前时间在今天的重置时间之前，则最近一次重置点是昨天的重置时间
    if now_bjt < today_reset_bjt:
        last_reset_bjt = today_reset_bjt - timedelta(days=1)
    else:
        # 否则，最近一次重置点是今天的重置时间
        last_reset_bjt = today_reset_bjt
        
    logging.info(f"检查重置：最近一次重置时间点为: {last_reset_bjt.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    reset_count = 0
    pool_modified = False
    
    for item in api_pool_data:
        if item.get('status') == '不可用':
            last_used_str = item.get('last_used')
            if not last_used_str:
                # 如果缺少 last_used 字段，为了安全，也跳过重置
                logging.warning(f"密钥 {item.get('alias', '无别称')} 缺少 last_used 字段，跳过检查。")
                continue
                
            last_used_bjt = parse_iso_time(last_used_str)
            
            # 3. 核心判断：只有当最后使用时间早于重置边界时才重置
            if last_used_bjt and last_used_bjt < last_reset_bjt:
                item['status'] = '可用'
                # 记录重置时间
                item['last_used'] = now_bjt.isoformat() 
                reset_count += 1
                pool_modified = True

    if reset_count > 0:
        logging.info(f"✅ 已根据 '{last_reset_bjt.strftime('%H:%M')}' 重置规则，共重置 {reset_count} 个密钥为 '可用'。")
    
    return pool_modified

def notify_server_for_update():
    """
    向后端服务发送 HTTP POST 请求，模拟前端的“保存所有设置”操作，
    触发后端的安全保存和内存热加载逻辑。
    """
    update_url = SERVER_URL + UPDATE_ENDPOINT
    
    # 1. 重新读取磁盘上最新的配置 (包含新的 API Key)
    # 这里使用我们之前定义的 CONFIG_JSON_PATH
    try:
        with open(CONFIG_JSON_PATH, "r", encoding="utf-8") as f:
            latest_config = json.load(f)
    except Exception as e:
        # 使用Flask的logging，或者直接print
        app.logger.error(f"无法读取 {CONFIG_JSON_PATH} 文件，无法通知后端服务更新。错误: {e}")
        return False

    # 2. 构造符合 FastAPI /api/update_llm_config 端点要求的 Payload
    payload = {"config": latest_config}

    try:
        # 使用超时避免长时间阻塞
        response = requests.post(update_url, json=payload, timeout=5)
        
        if response.status_code == 200:
            try:
                response_json = response.json()
                if response_json.get('status') == 'success':
                    app.logger.info("⭐ 已成功通知后端服务更新并热加载配置。")
                    return True
                else:
                    app.logger.error(f"后端服务响应失败状态: {response_json.get('message', '未知错误')}")
                    return False
            except json.JSONDecodeError:
                 app.logger.error(f"后端服务响应成功，但 JSON 解析失败。响应文本: {response.text[:100]}...")
                 return False
        else:
            app.logger.error(f"通知后端服务失败，HTTP 状态码: {response.status_code}, 响应: {response.text}")
            return False
    except requests.exceptions.ConnectionError:
        app.logger.error(f"连接到后端服务失败 ({update_url})，请确保服务正在运行。")
        return False
    except requests.exceptions.Timeout:
        app.logger.error("连接后端服务超时。")
        return False
    except Exception as e:
        app.logger.error(f"通知后端服务时发生未知错误: {e}")
        return False

# ----------------- 核心轮换逻辑 -----------------

def rotate_api_key():
    """
    执行 API 密钥轮换逻辑，精确针对 config.json 中的 'models' -> 'gemini' 路径。
    """
    
    # 1. 加载所有 JSON 文件
    config_data = load_json(CONFIG_JSON_PATH)
    api_pool_data = load_json(API_POOL_FILE_PATH)
    
    if config_data is None or api_pool_data is None:
        logging.error("加载配置或 API 池文件失败，跳过轮换。")
        return

    # 2. 检查并重置密钥状态 (必须先执行，以确保有可用的密钥)
    check_and_reset_keys(api_pool_data)
    # 即使重置了，也继续执行轮换（如果日志失败次数达到阈值）

    # 3. 获取当前密钥（旧密钥）信息
    try:
        current_key = config_data[TARGET_SECTION][TARGET_MODEL]["api_key"]
    except KeyError:
        logging.error(f"config.json 中未找到目标路径: {TARGET_SECTION} -> {TARGET_MODEL} -> api_key。请检查 config.json 结构。")
        return

    now_bjt_str = get_beijing_time().isoformat()
    old_key_item = next((item for item in api_pool_data if item.get('key') == current_key), None)
    old_alias = "未记录别称"

    # 4. 既然轮换被触发（因为失败），先将旧密钥标记为 '不可用'
    if old_key_item:
        old_alias = old_key_item.get('alias', "无别称")
        old_key_item['status'] = '不可用'
        old_key_item['last_used'] = now_bjt_str # 记录禁用时间
        logging.info(f"触发轮换：当前使用的密钥 ({old_alias}) 已被标记为 '不可用'。")
    else:
        # 如果当前密钥不在池中，将其添加并标记为不可用
        api_pool_data.append({
            "key": current_key,
            "alias": "运行时替换下来的密钥 (不在原始池中)",
            "status": "不可用",
            "last_used": now_bjt_str
        })
        old_alias = "运行时替换下来的密钥 (不在原始池中)"


    # 5. 查找下一个可用的且不是当前正在使用的密钥作为新密钥
    # 查找所有可用密钥中，键值不等于当前失败密钥的第一个密钥
    next_key_item = next((item for item in api_pool_data 
                          if item.get('status') == '可用' and item.get('key') != current_key), None)

    if not next_key_item:
        # 如果没有找到任何可用的替换密钥
        logging.critical(f"🚨 已无可用替换密钥！当前密钥 ({old_alias}) 已失效。请补充api.json中的密钥或等待重置时间。 🚨")
        
        # 保存 api_pool_data 的更改（即旧密钥被标记为不可用）
        if not save_json(API_POOL_FILE_PATH, api_pool_data):
             logging.error("保存 API 密钥池文件失败。")
        return

    # 6. 执行轮换操作
    new_key = next_key_item.get("key")
    new_key_alias = next_key_item.get("alias", "无别称")

    # 更新 config.json 中的密钥
    config_data[TARGET_SECTION][TARGET_MODEL]["api_key"] = new_key
    
    if save_json(CONFIG_JSON_PATH, config_data):
        
        # 更新 api_pool.json: 标记新密钥为 '不可用' (因为它现在正在被使用)
        next_key_item['status'] = '正在使用'
        next_key_item['last_used'] = now_bjt_str # 记录启用时间
        
        if save_json(API_POOL_FILE_PATH, api_pool_data):
            
            logging.info(f"✅ 密钥轮换成功！(目标: {TARGET_SECTION}.{TARGET_MODEL}.api_key)")
            logging.info(f"    旧密钥: {current_key[:10]}... ({old_alias}) 已被禁用")
            logging.info(f"    新密钥: {new_key[:10]}... ({new_key_alias}) 已启用")
            logging.info(f"    时间: {now_bjt_str}")
            
            # 7. 自动通知后端服务更新配置
            notify_server_for_update()
            
        else:
            logging.error("保存 API 密钥池文件失败，请手动检查文件状态。")
    else:
        logging.error("保存 config.json 失败，API 密钥未更新。")

def main_loop():
    """主循环，每隔一段时间执行一次检查"""
    logging.info(f"--- 密钥轮换监控脚本启动 (目标: {TARGET_SECTION}.{TARGET_MODEL}.api_key) ---")
    logging.info(f"检查间隔: {INTERVAL_SECONDS} 秒 ({INTERVAL_SECONDS / 60} 分钟)")
    
    while True:
        try:
            now_bjt = get_beijing_time()
            fail_count = get_log_fail_count(LOG_FILE_PATH)
            
            # 1. 加载配置和 API 池数据
            config_data = load_json(CONFIG_JSON_PATH)
            api_pool_data = load_json(API_POOL_FILE_PATH)
            
            can_rotate = True
            
            # 2. 密钥保护机制检查：检查当前密钥是否已启用至少 1 小时
            if config_data and api_pool_data:
                try:
                    current_key = config_data[TARGET_SECTION][TARGET_MODEL]["api_key"]
                    current_key_item = next((item for item in api_pool_data if item.get('key') == current_key), None)
                    
                    if current_key_item and current_key_item.get('last_used'):
                        last_used_bjt = parse_iso_time(current_key_item['last_used'])
                        
                        if last_used_bjt:
                            time_elapsed = now_bjt - last_used_bjt
                            one_hour = timedelta(hours=0.5)
                            
                            # 如果启用时间小于 1 小时，则阻止轮换
                            if time_elapsed < one_hour:
                                can_rotate = False
                                elapsed_minutes = int(time_elapsed.total_seconds() // 60)
                                logging.info(f"🔑 密钥保护机制触发：当前密钥已启用 {elapsed_minutes} 分钟，不足 1 小时 ({one_hour}). 暂不触发轮换。")
                            else:
                                elapsed_hours = time_elapsed.total_seconds() / 3600
                                logging.info(f"密钥已启用 {elapsed_hours:.2f} 小时。允许轮换。")
                        else:
                            # 密钥在池中但无启用时间，可能是旧密钥，默认允许轮换
                            logging.info("无法解析当前密钥的启用时间。跳过 1 小时保护检查。")

                    else:
                        # 密钥不在池中或无 'last_used' 字段，默认允许轮换
                        logging.info("无法在密钥池中找到当前密钥的启用信息。跳过 1 小时保护检查。")

                except KeyError:
                    logging.error(f"config.json 结构错误，无法获取当前密钥。")
                    can_rotate = False # 结构错误，阻止轮换

            # 3. 检查失败计数，并结合保护机制决定是否轮换
            if fail_count >= THRESHOLD:
                if can_rotate:
                    logging.warning(f"❗ 失败次数 ({fail_count}) 达到或超过阈值 ({THRESHOLD})，触发密钥轮换。")
                    rotate_api_key()
                else:
                    logging.warning(f"❗ 失败次数 ({fail_count}) 达到或超过阈值 ({THRESHOLD})，但密钥保护机制阻止了轮换。")
            else:
                logging.info("未达到轮换阈值，继续监控...")
                
            # 4. 检查并执行每日重置（重用 api_pool_data）
            if api_pool_data is not None and check_and_reset_keys(api_pool_data):
                # 如果发生了重置，需要保存 pool 文件
                save_json(API_POOL_FILE_PATH, api_pool_data)


        except Exception as e:
            logging.error(f"主循环中发生未预期的错误: {e}")
            
        # 休息一段时间
        time.sleep(INTERVAL_SECONDS)

# --- 全局批量认证（必须在app实例创建后定义）---
@app.before_request
def global_auth():
    # 排除不需要保护的静态资源和特殊API路径
    excluded_paths = ["/favicon.ico", "/update_indextts_endpoint"]
    if request.path in excluded_paths:
        return  # 如果是排除的路径，直接放行，不做任何处理
    
    # 所有其他请求强制认证
    auth = request.authorization
    if not auth or not check_auth(auth.username, auth.password):
        return authenticate()


# --- 通用文件读取函数 ---
def read_file_content(file_path, max_lines=200):
    """通用文件读取函数，支持文本文件和JSON文件"""
    content = []
    file_type = os.path.splitext(file_path)[1].lower()
    
    if not os.path.exists(file_path):
        return [{"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                 "content": f"文件不存在（路径：{file_path}）"}]
    
    try:
        if file_type == '.json':
            # 处理JSON文件
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                json_data = json.load(f)
                # 将JSON数据格式化显示
                content.append({"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                               "content": "=== JSON 文件内容 ==="})
                content.append({"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                               "content": json.dumps(json_data, indent=2, ensure_ascii=False)})
        else:
            # 处理日志文件
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                all_lines = f.readlines()
                lines = all_lines[-max_lines:] if len(all_lines) > max_lines else all_lines
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    log_time = line.split(" - ")[0] if " - " in line else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    content.append({"time": log_time, "content": line})
                    
    except Exception as e:
        content = [{"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                   "content": f"读取文件失败: {str(e)}"}]
    
    return content

# --- 路由定义 ---
@app.route("/")
def index():
    return redirect(url_for('log_page', file_type='output'))

@app.route("/logs/<file_type>")
def log_page(file_type):
    return render_template("log_monitor.html", file_type=file_type)

@app.route("/get_logs")
def get_logs():
    file_type = request.args.get('file_type', 'output')
    
    if file_type == 'api':
        logs = read_file_content(API_JSON_PATH)
    elif file_type == 'key':
        logs = read_file_content(KEY_LOG_PATH)
    else:
        logs = read_file_content(LOG_FILE_PATH)
    
    return jsonify({"logs": logs, "total": len(logs)})

@app.route("/save_api_data", methods=["POST"])
def save_api_data():
    try:
        data = request.json.get("data", [])
        with open(API_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return jsonify({"success": True, "message": "保存成功"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/update_indextts_endpoint", methods=["GET"])
def update_indextts_endpoint():
    """
    通过GET参数更新config.json中indextts_v2的endpoint域名，并通知后端热加载。
    例如: /update_indextts_endpoint?new_host=https://new.domain.com
    """
    # --- Token验证 ---
    provided_token = request.args.get('token')
    print(f"DEBUG: Received token: '{provided_token}'") # <--- 调试信息
    if provided_token != API_TOKEN:
        print(f"DEBUG: Token mismatch! Expected: '{API_TOKEN}'") # <--- 调试信息
        return jsonify({"success": False, "message": "Invalid or missing API token."}), 401

    new_host = request.args.get('new_host')

    if not new_host:
        return jsonify({
            "success": False,
            "message": "缺少必要参数 'new_host'。请使用 ?new_host=你的新域名"
        }), 400

    if not os.path.exists(CONFIG_JSON_PATH):
        return jsonify({
            "success": False,
            "message": f"配置文件不存在: {CONFIG_JSON_PATH}"
        }), 404

    try:
        with open(CONFIG_JSON_PATH, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        tts_models = config_data.get("tts_models", {})
        indextts_config = tts_models.get("indextts_v2", {})
        current_endpoint = indextts_config.get("endpoint", "")

        if not current_endpoint:
            return jsonify({
                "success": False,
                "message": "配置文件中未找到 'tts_models.indextts_v2.endpoint'"
            }), 404

        from urllib.parse import urlparse, urlunparse
        parsed_url = urlparse(current_endpoint)
        # 确保新主机名不包含协议，并且路径不以/开头，避免双斜杠
        clean_new_host = new_host.replace("https://" , "").replace("http://"
        "http://"
         , "")
        clean_path = parsed_url.path.lstrip('/')

        new_endpoint = urlunparse((
            parsed_url.scheme,      # 协议，如 https
            clean_new_host,         # 清理后的新域名
            '/' + clean_path,       # 确保路径以/开头
            parsed_url.params,      # 参数
            parsed_url.query,       # 查询字符串
            parsed_url.fragment     # 片段
        ))

        config_data["tts_models"]["indextts_v2"]["endpoint"] = new_endpoint

        with open(CONFIG_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)

        # --- 关键改动：保存文件后，立即调用函数通知后端 ---
        notification_success = notify_server_for_update()

        if notification_success:
            return jsonify({
                "success": True,
                "message": "Endpoint更新成功，并已通知后端热加载！",
                "new_endpoint": new_endpoint
            })
        else:
            # 即使通知失败，文件也已更新，但需要告知用户
            return jsonify({
                "success": True,
                "message": "Endpoint更新成功，但通知后端热加载失败，请检查后端服务日志。",
                "new_endpoint": new_endpoint,
                "warning": "Backend notification failed."
            }), 202 # 使用 202 Accepted 状态码表示操作已接受但处理未完成

    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"更新配置时发生错误: {str(e)}"
        }), 500

# 在 log.py 中添加新的路由
@app.route("/rotate_key", methods=["POST"])
def trigger_key_rotation():
    """
    手动触发API密钥轮换的接口。
    需要Basic Auth认证。
    """
    try:
        # 直接调用从 key.py 复制过来的核心轮换函数
        rotate_api_key()
        return jsonify({
            "success": True,
            "message": "密钥轮换任务已成功触发并执行。请查看日志以获取详细信息。"
        })
    except Exception as e:
        app.logger.error(f"触发密钥轮换时发生错误: {e}")
        return jsonify({
            "success": False,
            "message": f"触发密钥轮换失败: {str(e)}"
        }), 500
		
# 在 log.py 中找到 get_key_status 函数
@app.route("/get_key_status", methods=["GET"])
def get_key_status():
    try:
        api_pool_data = load_json(API_POOL_FILE_PATH)
        if api_pool_data is None:
            return jsonify({"success": False, "message": "无法加载API密钥池文件。"}), 500

        config_data = load_json(CONFIG_JSON_PATH)
        current_key = "未知"
        if config_data:
            try:
                current_key = config_data[TARGET_SECTION][TARGET_MODEL]["api_key"]
            except KeyError:
                pass

        # --- 新增逻辑：动态标记当前正在使用的密钥 ---
        for item in api_pool_data:
            if item.get('key') == current_key:
                item['status'] = '正在使用' # 动态设置为“正在使用”
                break
        # ------------------------------------------------

        current_key_info = next((item for item in api_pool_data if item.get('key') == current_key), None)

        return jsonify({
            "success": True,
            "current_key_alias": current_key_info.get('alias') if current_key_info else "当前密钥不在池中",
            "current_key_last_used": current_key_info.get('last_used') if current_key_info else None,
            "total_keys": len(api_pool_data),
            "available_keys": len([k for k in api_pool_data if k.get('status') == '可用']),
            "unavailable_keys": len([k for k in api_pool_data if k.get('status') == '不可用']),
            "in_use_keys": len([k for k in api_pool_data if k.get('status') == '正在使用']), # 新增统计
            "pool_details": api_pool_data
        })
    except Exception as e:
        app.logger.error(f"获取密钥状态时发生错误: {e}")
        return jsonify({
            "success": False,
            "message": f"获取密钥状态失败: {str(e)}"
        }), 500

# --- 在 log.py 中添加新的修改状态接口 ---
@app.route("/update_key_status", methods=["POST"])
def update_key_status():
    """
    修改单个密钥的状态。
    需要Basic Auth认证。
    """
    try:
        data = request.json
        key_to_update = data.get('key')
        new_status = data.get('status')

        if not key_to_update or not new_status:
            return jsonify({"success": False, "message": "缺少必要参数 'key' 或 'status'。"}), 400

        if new_status not in ['可用', '不可用']:
            return jsonify({"success": False, "message": "无效的状态值。只允许 '可用' 或 '不可用'。"}), 400

        api_pool_data = load_json(API_POOL_FILE_PATH)
        if not api_pool_data:
            return jsonify({"success": False, "message": "无法加载API密钥池文件。"}), 500

        # 找到并更新密钥
        key_updated = False
        for item in api_pool_data:
            if item.get('key') == key_to_update:
                item['status'] = new_status
                key_updated = True
                break
        
        if not key_updated:
            return jsonify({"success": False, "message": f"未找到密钥: {key_to_update[:10]}..."}), 404

        if save_json(API_POOL_FILE_PATH, api_pool_data):
            return jsonify({"success": True, "message": "密钥状态更新成功。"})
        else:
            return jsonify({"success": False, "message": "保存文件失败。"}), 500

    except Exception as e:
        app.logger.error(f"更新密钥状态时发生错误: {e}")
        return jsonify({"success": False, "message": f"更新失败: {str(e)}"}), 500


# --- 在 log.py 中添加新的新增密钥接口 ---
@app.route("/add_new_key", methods=["POST"])
def add_new_key():
    """
    向API密钥池中添加一个新密钥。
    需要Basic Auth认证。
    """
    try:
        data = request.json
        new_key = data.get('key')
        new_alias = data.get('alias')

        if not new_key or not new_alias:
            return jsonify({"success": False, "message": "缺少必要参数 'key' 或 'alias'。"}), 400

        api_pool_data = load_json(API_POOL_FILE_PATH)
        if not api_pool_data:
            return jsonify({"success": False, "message": "无法加载API密钥池文件。"}), 500

        # 检查密钥是否已存在
        if any(item.get('key') == new_key for item in api_pool_data):
            return jsonify({"success": False, "message": "该密钥已存在于密钥池中。"}), 409

        # 添加新密钥，默认状态为 '可用'
        new_entry = {
            "key": new_key,
            "alias": new_alias,
            "status": "可用",
            "last_used": None
        }
        api_pool_data.append(new_entry)

        if save_json(API_POOL_FILE_PATH, api_pool_data):
            return jsonify({"success": True, "message": "新密钥添加成功。"})
        else:
            return jsonify({"success": False, "message": "保存文件失败。"}), 500

    except Exception as e:
        app.logger.error(f"添加新密钥时发生错误: {e}")
        return jsonify({"success": False, "message": f"添加失败: {str(e)}"}), 500

# --- 在 log.py 中添加新的删除密钥接口 ---
@app.route("/delete_key", methods=["POST"])
def delete_key():
    """
    从API密钥池中删除一个密钥。
    需要Basic Auth认证。
    """
    try:
        data = request.json
        key_to_delete = data.get('key')

        if not key_to_delete:
            return jsonify({"success": False, "message": "缺少必要参数 'key'。"}), 400

        api_pool_data = load_json(API_POOL_FILE_PATH)
        if not api_pool_data:
            return jsonify({"success": False, "message": "无法加载API密钥池文件。"}), 500

        # 检查密钥是否正在被使用
        config_data = load_json(CONFIG_JSON_PATH)
        current_key = "未知"
        if config_data:
            try:
                current_key = config_data[TARGET_SECTION][TARGET_MODEL]["api_key"]
            except KeyError:
                pass
        
        if key_to_delete == current_key:
            return jsonify({"success": False, "message": "不能删除当前正在使用的密钥！"}), 409

        # 找到并删除密钥
        initial_length = len(api_pool_data)
        api_pool_data[:] = [item for item in api_pool_data if item.get('key') != key_to_delete]

        if len(api_pool_data) == initial_length:
            return jsonify({"success": False, "message": f"未找到密钥: {key_to_delete[:10]}..."}), 404

        if save_json(API_POOL_FILE_PATH, api_pool_data):
            return jsonify({"success": True, "message": "密钥删除成功。"})
        else:
            return jsonify({"success": False, "message": "保存文件失败。"}), 500

    except Exception as e:
        app.logger.error(f"删除密钥时发生错误: {e}")
        return jsonify({"success": False, "message": f"删除失败: {str(e)}"}), 500


@app.errorhandler(404)
def page_not_found(e):
    return redirect(url_for('index'))

if __name__ == "__main__":
    PORT = 8888
    print(f"日志监控服务启动：http://0.0.0.0:{PORT}")
    print(f"监控文件列表：")
    print(f"  - Output日志: {LOG_FILE_PATH}")
    print(f"  - API JSON: {API_JSON_PATH}")
    print(f"  - Key日志: {KEY_LOG_PATH}")
    print(f"认证信息：用户名={USERNAME}，密码={PASSWORD}")
    print(f"访问路径：")
    print(f"  - http://0.0.0.0:{PORT}/logs/output (默认日志)")
    print(f"  - http://0.0.0.0:{PORT}/logs/api (API JSON文件)")
    print(f"  - http://0.0.0.0:{PORT}/logs/key (Key日志文件)")
    app.run(host="0.0.0.0", port=PORT, debug=False)