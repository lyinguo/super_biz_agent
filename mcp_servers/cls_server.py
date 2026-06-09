"""腾讯云 CLS (Cloud Log Service) MCP Server

本地实现的 CLS 日志服务 MCP Server，提供日志查询、检索和分析功能。
"""

import logging
import functools
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from fastmcp import FastMCP

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CLS_MCP_Server")

mcp = FastMCP("CLS")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_SUFFIXES = {".log", ".out", ".err"}
EXCLUDED_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "volumes",
    "memory_summaries",
}
MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_SCAN_LINES_PER_FILE = 3000

ANOMALY_RULES = [
    (
        "critical",
        re.compile(r"\b(CRITICAL|FATAL|PANIC)\b|严重|致命", re.IGNORECASE),
    ),
    (
        "error",
        re.compile(
            r"\b(ERROR|ERR|Exception|Traceback|failed|failure|timeout|refused|denied)\b"
            r"|错误|异常|失败|超时|拒绝",
            re.IGNORECASE,
        ),
    ),
    (
        "warning",
        re.compile(r"\b(WARN|WARNING)\b|警告|告警", re.IGNORECASE),
    ),
]
TIMESTAMP_PATTERN = re.compile(r"(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})")
LOGURU_LEVEL_PATTERN = re.compile(r"\|\s*(?P<level>[A-Z]+)\s*\|")


def log_tool_call(func):
    """装饰器：记录工具调用的日志，包括方法名、参数和返回状态"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        method_name = func.__name__

        # 记录调用信息
        logger.info(f"=" * 80)
        logger.info(f"调用方法: {method_name}")

        # 记录参数（排除self等）
        if kwargs:
            # 使用 json.dumps 格式化参数，处理可能的序列化错误
            try:
                params_str = json.dumps(kwargs, ensure_ascii=False, indent=2)
            except (TypeError, ValueError):
                params_str = str(kwargs)
            logger.info(f"参数信息:\n{params_str}")
        else:
            logger.info("参数信息: 无")

        # 执行方法
        try:
            result = func(*args, **kwargs)

            # 记录返回状态
            logger.info(f"返回状态: SUCCESS")

            # 记录返回结果摘要（避免日志过长）
            if isinstance(result, dict):
                summary = {k: v if not isinstance(v, (list, dict)) else f"<{type(v).__name__} with {len(v)} items>"
                          for k, v in list(result.items())[:5]}
                logger.info(f"返回结果摘要: {json.dumps(summary, ensure_ascii=False)}")
            else:
                logger.info(f"返回结果: {result}")

            logger.info(f"=" * 80)
            return result

        except Exception as e:
            # 记录错误状态
            logger.error(f"返回状态: ERROR")
            logger.error(f"错误信息: {str(e)}")
            logger.error(f"=" * 80)
            raise

    return wrapper


def parse_time_or_default(time_str: Optional[str], default_offset_hours: int = 0) -> datetime:
    """解析时间字符串或返回默认时间。

    Args:
        time_str: 时间字符串（格式：YYYY-MM-DD HH:MM:SS）
        default_offset_hours: 默认时间偏移（小时）

    Returns:
        datetime: 解析后的时间对象
    """
    if time_str:
        try:
            return datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
    return datetime.now() + timedelta(hours=default_offset_hours)


def generate_time_series(base_time: datetime, minutes_offset: int) -> str:
    """生成基于基准时间的时间字符串。

    Args:
        base_time: 基准时间
        minutes_offset: 分钟偏移量

    Returns:
        str: 格式化的时间字符串
    """
    result_time = base_time + timedelta(minutes=minutes_offset)
    return result_time.strftime("%Y-%m-%d %H:%M:%S")


def is_safe_project_path(path: Path) -> bool:
    """限制日志扫描只访问当前项目目录内的文件。"""
    try:
        path.resolve().relative_to(PROJECT_ROOT)
        return True
    except ValueError:
        return False


def should_skip_path(path: Path) -> bool:
    """跳过虚拟环境、数据库数据目录等不适合作为业务日志扫描的路径。"""
    return any(part in EXCLUDED_DIR_NAMES for part in path.parts)


def discover_local_log_files() -> list[Path]:
    """发现当前项目下可扫描的本地日志文件。"""
    candidates: list[Path] = []

    # 优先扫描项目 logs 目录，再扫描项目根目录的一层日志文件。
    scan_roots = [PROJECT_ROOT / "logs", PROJECT_ROOT]
    for root in scan_roots:
        if not root.exists() or not root.is_dir():
            continue

        iterator = root.rglob("*") if root.name == "logs" else root.glob("*")
        for path in iterator:
            if not path.is_file():
                continue
            if should_skip_path(path):
                continue
            if path.suffix.lower() not in LOG_SUFFIXES:
                continue
            if not is_safe_project_path(path):
                continue
            candidates.append(path.resolve())

    # 去重并按最近修改时间排序，最新日志排在前面。
    unique = sorted(set(candidates), key=lambda p: p.stat().st_mtime, reverse=True)
    return unique


def topic_id_for_path(path: Path) -> str:
    relative_path = path.resolve().relative_to(PROJECT_ROOT).as_posix()
    return f"local-log:{relative_path}"


def path_for_topic_id(topic_id: str) -> Path | None:
    """根据 topic_id 找到真实日志文件；兼容旧 topic-001，映射到最新日志。"""
    files = discover_local_log_files()
    if topic_id in {"topic-001", "local", "latest"}:
        return files[0] if files else None

    if topic_id.startswith("local-log:"):
        relative = topic_id.removeprefix("local-log:")
        path = (PROJECT_ROOT / relative).resolve()
        if path.exists() and path.is_file() and is_safe_project_path(path):
            return path

    return None


def infer_service_name(path: Path) -> str:
    """从日志文件名推断一个粗略的服务名，供 Agent 选择 topic。"""
    stem = path.stem.lower()
    if stem.startswith("app"):
        return "super-biz-agent"
    if "server" in stem:
        return "fastapi-server"
    if "mcp" in stem:
        return "mcp-server"
    return stem.replace("_", "-")


def file_topic(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    relative_path = path.relative_to(PROJECT_ROOT).as_posix()
    return {
        "topic_id": topic_id_for_path(path),
        "topic_name": path.name,
        "service_name": infer_service_name(path),
        "region_code": "local",
        "create_time": datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S"),
        "update_time": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "size_bytes": stat.st_size,
        "path": relative_path,
        "description": "本项目本地日志文件",
    }


def parse_log_timestamp(line: str) -> datetime | None:
    match = TIMESTAMP_PATTERN.search(line)
    if not match:
        return None
    timestamp_text = match.group("ts").replace("T", " ")
    try:
        return datetime.strptime(timestamp_text, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def detect_log_level(line: str, severity: str) -> str:
    match = LOGURU_LEVEL_PATTERN.search(line)
    if match:
        return match.group("level")
    upper = line.upper()
    for level in ("CRITICAL", "ERROR", "WARNING", "WARN", "INFO", "DEBUG"):
        if level in upper:
            return "WARNING" if level == "WARN" else level
    return severity.upper() if severity != "normal" else "INFO"


def classify_anomaly(line: str) -> tuple[bool, str, list[str]]:
    matched_keywords: list[str] = []
    for severity, pattern in ANOMALY_RULES:
        matches = pattern.findall(line)
        if matches:
            for item in matches:
                if isinstance(item, tuple):
                    matched_keywords.extend(str(x) for x in item if x)
                else:
                    matched_keywords.append(str(item))
            return True, severity, sorted(set(matched_keywords))
    return False, "normal", []


def read_recent_log_lines(path: Path, max_lines: int = MAX_SCAN_LINES_PER_FILE) -> list[tuple[int | None, str]]:
    """读取日志尾部内容；大文件只读取最后 MAX_FILE_BYTES，避免工具调用过慢。"""
    size = path.stat().st_size
    if size <= MAX_FILE_BYTES:
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        start_line = max(1, len(lines) - max_lines + 1)
        return list(enumerate(lines[-max_lines:], start=start_line))

    with path.open("rb") as f:
        f.seek(max(0, size - MAX_FILE_BYTES))
        if f.tell() > 0:
            f.readline()
        data = f.read()
    lines = data.decode("utf-8", errors="replace").splitlines()
    return [(None, line) for line in lines[-max_lines:]]


def line_matches_query(line: str, level: str, query: Optional[str]) -> bool:
    if not query:
        return True

    query_text = query.strip()
    if not query_text:
        return True

    lower_line = line.lower()
    lower_query = query_text.lower()

    # 支持常见的简单形式：level:ERROR、message:timeout，以及 OR 组合。
    parts = [p.strip() for p in re.split(r"\s+OR\s+", query_text, flags=re.IGNORECASE) if p.strip()]
    if len(parts) > 1:
        return any(line_matches_query(line, level, part) for part in parts)

    if lower_query.startswith("level:"):
        expected = query_text.split(":", 1)[1].strip().upper()
        return expected in level.upper() or expected in line.upper()

    if lower_query.startswith("message:"):
        expected = query_text.split(":", 1)[1].strip().lower()
        return expected in lower_line

    return lower_query in lower_line


def in_time_range(log_time: datetime | None, start_time: int, end_time: int) -> bool:
    if log_time is None:
        return True
    timestamp_ms = int(log_time.timestamp() * 1000)
    return start_time <= timestamp_ms <= end_time


@mcp.tool()
@log_tool_call
def get_current_timestamp() -> int:
    """获取当前时间戳（以毫秒为单位）。
    
    此工具用于获取标准的毫秒时间戳，可用于：
    1. 作为 search_log 的 end_time 参数（查询到现在）
    2. 计算历史时间点作为 start_time 参数
    
    Returns:
        int: 当前时间戳（毫秒），例如: 1708012345000
    
    使用示例:
        # 获取当前时间
        current = get_current_timestamp()
        
        # 计算15分钟前的时间
        fifteen_min_ago = current - (15 * 60 * 1000)
        
        # 计算1小时前的时间
        one_hour_ago = current - (60 * 60 * 1000)
        
        # 用于搜索最近15分钟的日志
        search_log(
            topic_id="topic-001",
            start_time=fifteen_min_ago,
            end_time=current
        )
    """
    return int(datetime.now().timestamp() * 1000)


@mcp.tool()
@log_tool_call
def get_region_code_by_name(region_name: str) -> Dict[str, Any]:
    """根据地区名称搜索对应的地区参数。

    Args:
        region_name: 地区名称（如：北京、上海、广州等）

    Returns:
        Dict: 包含地区代码和相关信息的字典
            - region_code: 地区代码
            - region_name: 地区名称
            - available: 是否可用
    """
    # 模拟地区映射表（实际应该从配置或数据库读取）
    region_mapping = {
        "北京": {"region_code": "ap-beijing", "region_name": "北京", "available": True},
        "上海": {"region_code": "ap-shanghai", "region_name": "上海", "available": True},
        "广州": {"region_code": "ap-guangzhou", "region_name": "广州", "available": True},
    }

    result = region_mapping.get(region_name)
    if result:
        return result
    else:
        return {
            "region_code": None,
            "region_name": region_name,
            "available": False,
            "error": f"未找到地区: {region_name}"
        }


@mcp.tool()
@log_tool_call
def get_topic_info_by_name(topic_name: str, region_code: Optional[str] = None) -> Dict[str, Any]:
    """根据主题名称搜索相关的主题信息。

    Args:
        topic_name: 主题名称
        region_code: 地区代码（可选）

    Returns:
        Dict: 包含主题信息的字典
            - topic_id: 主题ID
            - topic_name: 主题名称
            - region_code: 所属地区
            - create_time: 创建时间
            - log_count: 日志数量
    """
    topics = [file_topic(path) for path in discover_local_log_files()]

    for topic in topics:
        if region_code and topic["region_code"] != region_code:
            continue
        if topic["topic_name"] == topic_name or topic["path"] == topic_name:
            return topic

    return {
        "topic_id": None,
        "topic_name": topic_name,
        "region_code": region_code,
        "error": f"未找到本地日志主题: {topic_name}",
        "available_topics": topics[:10],
    }


@mcp.tool()
@log_tool_call
def search_topic_by_service_name(
    service_name: str,
    region_code: Optional[str] = None,
    fuzzy: bool = True
) -> Dict[str, Any]:
    """根据服务名称搜索相关的日志主题信息，支持模糊搜索。
    
    此工具用于根据服务名称查找对应的日志主题（topic），便于后续进行日志查询。
    
    Args:
        service_name: 服务名称（必填）
            示例: "data-sync-service", "sync", "data-sync"
            说明: 当 fuzzy=True 时，支持部分匹配
        
        region_code: 地区代码（可选）
            示例: "ap-beijing", "ap-shanghai"
            说明: 如果指定，只返回该地区的主题
        
        fuzzy: 是否启用模糊搜索（可选，默认 True）
            True: 部分匹配，例如 "sync" 可以匹配 "data-sync-service"
            False: 精确匹配，必须完全一致
    
    Returns:
        Dict: 搜索结果
            - total: 匹配到的主题数量
            - topics: 主题列表，每个主题包含:
                * topic_id: 主题ID（用于后续日志查询）
                * topic_name: 主题名称
                * service_name: 服务名称
                * region_code: 所属地区
                * create_time: 创建时间
                * log_count: 日志数量
                * description: 主题描述
            - query: 查询条件
    
    使用示例:
        # 示例1: 模糊搜索（推荐）
        search_topic_by_service_name(service_name="data-sync")
        # 可以匹配: "data-sync-service", "data-sync-worker" 等
        
        # 示例2: 精确搜索
        search_topic_by_service_name(
            service_name="data-sync-service",
            fuzzy=False
        )
        
        # 示例3: 指定地区搜索
        search_topic_by_service_name(
            service_name="sync",
            region_code="ap-beijing"
        )
        
        # 示例4: 查找后进行日志搜索的完整流程
        # 步骤1: 根据服务名查找 topic
        result = search_topic_by_service_name(service_name="data-sync-service")
        
        # 步骤2: 获取 topic_id
        topic_id = result["topics"][0]["topic_id"]  # "topic-001"
        
        # 步骤3: 使用 topic_id 查询日志
        current_ts = get_current_timestamp()
        start_ts = current_ts - (15 * 60 * 1000)
        search_log(
            topic_id=topic_id,
            start_time=start_ts,
            end_time=current_ts
        )
    """
    topics = [file_topic(path) for path in discover_local_log_files()]
    matched_topics = []

    for topic in topics:
        if region_code and topic["region_code"] != region_code:
            continue

        searchable_text = " ".join(
            [
                topic.get("service_name", ""),
                topic.get("topic_name", ""),
                topic.get("path", ""),
                topic.get("description", ""),
            ]
        ).lower()
        target = service_name.lower()

        if fuzzy:
            if target in searchable_text or any(part and part in searchable_text for part in target.split("-")):
                matched_topics.append(topic)
        else:
            if topic.get("service_name", "").lower() == target:
                matched_topics.append(topic)

    fallback_used = False
    if not matched_topics and fuzzy:
        # AIOps 可能传入业务服务名，但本地日志文件名未必包含该服务名。
        # 为了让诊断仍能看到真实日志，未匹配时回退返回最新的本地日志。
        matched_topics = topics[:5]
        fallback_used = bool(matched_topics)

    return {
        "total": len(matched_topics),
        "topics": matched_topics,
        "query": {
            "service_name": service_name,
            "region_code": region_code,
            "fuzzy": fuzzy,
            "source": "local_files",
            "project_root": str(PROJECT_ROOT),
            "fallback_used": fallback_used,
        },
        "message": (
            f"找到 {len(matched_topics)} 个本地日志主题"
            if matched_topics
            else f"未找到服务 '{service_name}' 可扫描的本地日志文件"
        ),
    }


@mcp.tool()
@log_tool_call
def search_log(
    topic_id: str,
    start_time: int,
    end_time: int,
    query: Optional[str] = None,
    limit: int = 100
) -> Dict[str, Any]:
    """基于提供的查询参数搜索日志。

    Args:
        topic_id: 主题ID（必填）
            示例: "topic-001"
        
        start_time: 开始时间戳，单位为毫秒（必填，int类型）
            重要: 必须传递整数类型的毫秒时间戳
            获取方式: 
            1. 使用 get_current_timestamp() 工具获取当前时间戳
            2. 计算历史时间: current_timestamp - (分钟数 * 60 * 1000)
            示例: 
            - 当前时间: 1708012345000
            - 15分钟前: 1708012345000 - (15 * 60 * 1000) = 1708011445000
            - 1小时前: 1708012345000 - (60 * 60 * 1000) = 1708008745000
        
        end_time: 结束时间戳，单位为毫秒（必填，int类型）
            重要: 必须传递整数类型的毫秒时间戳
            通常使用 get_current_timestamp() 工具获取当前时间作为结束时间
            示例: 1708012345000
        
        query: 查询语句（可选，CLS 查询语法）
            示例: "level:ERROR" 或 "message:异常"
        
        limit: 返回结果数量限制（默认100，可选）

    Returns:
        Dict: 搜索结果
            - topic_id: 主题ID
            - start_time: 开始时间戳
            - end_time: 结束时间戳
            - query: 查询语句
            - limit: 结果限制
            - total: 实际返回的日志条数
            - logs: 日志列表，每条日志包含:
                * timestamp: 日志时间（格式: YYYY-MM-DD HH:MM:SS）
                * level: 日志级别
                * message: 日志内容
            - took_ms: 查询耗时（毫秒）
            - message: 查询状态消息
    
    使用示例:
        # 步骤1: 获取当前时间戳
        current_ts = get_current_timestamp()  # 返回: 1708012345000
        
        # 步骤2: 计算开始时间（15分钟前）
        start_ts = current_ts - (15 * 60 * 1000)  # 1708011445000
        
        # 步骤3: 搜索日志
        search_log(
            topic_id="topic-001",
            start_time=start_ts,     # int类型: 1708011445000
            end_time=current_ts,     # int类型: 1708012345000
            limit=100
        )
    """
    started_at = datetime.now()
    path = path_for_topic_id(topic_id)
    if path is None:
        return {
            "topic_id": topic_id,
            "start_time": start_time,
            "end_time": end_time,
            "query": query,
            "limit": limit,
            "source": "local_files",
            "total": 0,
            "logs": [],
            "took_ms": 0,
            "error": f"本地日志主题不存在或不可访问: {topic_id}",
            "available_topics": [file_topic(p) for p in discover_local_log_files()[:10]],
            "message": f"错误: 未找到本地日志主题 {topic_id}",
        }

    limit = max(1, min(int(limit), 500))
    logs = []
    severity_counts: Dict[str, int] = {}
    scanned_lines = 0

    for line_number, line in read_recent_log_lines(path):
        scanned_lines += 1
        log_time = parse_log_timestamp(line)
        is_anomaly, severity, matched_keywords = classify_anomaly(line)
        level = detect_log_level(line, severity)

        if not in_time_range(log_time, start_time, end_time):
            continue
        if not line_matches_query(line, level, query):
            continue

        severity_counts[severity] = severity_counts.get(severity, 0) + 1
        timestamp_ms = int(log_time.timestamp() * 1000) if log_time else None
        logs.append(
            {
                "timestamp": log_time.strftime("%Y-%m-%d %H:%M:%S") if log_time else None,
                "timestamp_ms": timestamp_ms,
                "level": level,
                "message": line,
                "source_file": path.relative_to(PROJECT_ROOT).as_posix(),
                "line_number": line_number,
                "is_anomaly": is_anomaly,
                "severity": severity,
                "matched_keywords": matched_keywords,
            }
        )

    # 最新日志通常更有价值，返回尾部命中的 limit 条。
    selected_logs = logs[-limit:]
    anomaly_count = sum(1 for item in selected_logs if item["is_anomaly"])
    took_ms = int((datetime.now() - started_at).total_seconds() * 1000)

    return {
        "topic_id": topic_id,
        "topic": file_topic(path),
        "start_time": start_time,
        "end_time": end_time,
        "query": query,
        "limit": limit,
        "source": "local_files",
        "scan_scope": {
            "project_root": str(PROJECT_ROOT),
            "max_file_bytes": MAX_FILE_BYTES,
            "max_scan_lines_per_file": MAX_SCAN_LINES_PER_FILE,
            "scanned_lines": scanned_lines,
        },
        "total": len(selected_logs),
        "matched_total_before_limit": len(logs),
        "anomaly_count": anomaly_count,
        "severity_counts": severity_counts,
        "logs": selected_logs,
        "took_ms": took_ms,
        "message": f"成功从本地日志文件 {path.name} 查询 {len(selected_logs)} 条日志，异常 {anomaly_count} 条",
    }


@mcp.tool()
@log_tool_call
def scan_local_log_anomalies(
    hours: int = 24,
    limit_per_file: int = 50,
    include_normal: bool = False,
) -> Dict[str, Any]:
    """自动扫描当前项目本地日志文件，并标注异常日志。

    Args:
        hours: 扫描最近多少小时的日志，默认 24 小时。
        limit_per_file: 每个日志文件最多返回多少条，默认 50，最大 200。
        include_normal: 是否返回普通日志。默认 False，只返回异常/告警日志。

    Returns:
        Dict: 每个日志文件的异常统计、严重级别统计和命中的日志行。
    """
    started_at = datetime.now()
    hours = max(1, min(int(hours), 24 * 30))
    limit_per_file = max(1, min(int(limit_per_file), 200))
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(hours=hours)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    file_results = []
    global_severity_counts: Dict[str, int] = {}
    total_anomalies = 0
    total_returned = 0

    for path in discover_local_log_files():
        matched_logs = []
        file_severity_counts: Dict[str, int] = {}
        scanned_lines = 0

        for line_number, line in read_recent_log_lines(path):
            scanned_lines += 1
            log_time = parse_log_timestamp(line)
            if not in_time_range(log_time, start_ms, end_ms):
                continue

            is_anomaly, severity, matched_keywords = classify_anomaly(line)
            if not include_normal and not is_anomaly:
                continue

            level = detect_log_level(line, severity)
            file_severity_counts[severity] = file_severity_counts.get(severity, 0) + 1
            global_severity_counts[severity] = global_severity_counts.get(severity, 0) + 1

            matched_logs.append(
                {
                    "timestamp": log_time.strftime("%Y-%m-%d %H:%M:%S") if log_time else None,
                    "timestamp_ms": int(log_time.timestamp() * 1000) if log_time else None,
                    "level": level,
                    "message": line,
                    "source_file": path.relative_to(PROJECT_ROOT).as_posix(),
                    "line_number": line_number,
                    "is_anomaly": is_anomaly,
                    "severity": severity,
                    "matched_keywords": matched_keywords,
                }
            )

        selected_logs = matched_logs[-limit_per_file:]
        anomaly_count = sum(1 for item in matched_logs if item["is_anomaly"])
        total_anomalies += anomaly_count
        total_returned += len(selected_logs)

        file_results.append(
            {
                "topic": file_topic(path),
                "scanned_lines": scanned_lines,
                "matched_total_before_limit": len(matched_logs),
                "returned": len(selected_logs),
                "anomaly_count": anomaly_count,
                "severity_counts": file_severity_counts,
                "logs": selected_logs,
            }
        )

    took_ms = int((datetime.now() - started_at).total_seconds() * 1000)
    return {
        "success": True,
        "source": "local_files",
        "project_root": str(PROJECT_ROOT),
        "time_range": {
            "start": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "end": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "hours": hours,
        },
        "files_scanned": len(file_results),
        "total_anomalies": total_anomalies,
        "total_returned": total_returned,
        "severity_counts": global_severity_counts,
        "files": file_results,
        "took_ms": took_ms,
        "message": f"已扫描 {len(file_results)} 个本地日志文件，发现异常/告警 {total_anomalies} 条",
    }


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8003, path="/mcp")
