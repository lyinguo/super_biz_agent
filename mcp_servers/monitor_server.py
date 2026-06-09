"""智能运维监控 MCP Server

本地实现的监控服务 MCP Server，提供：
- 监控数据查询（CPU、内存、磁盘、网络等）
- 进程信息查询
- 历史工单查询
- 服务信息查询

用于支持运维 Agent 的故障排查场景。
"""

import logging
import functools
import json
import os
import platform
import socket
import time
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from fastmcp import FastMCP

try:
    import psutil
except ImportError:  # 让 MCP 服务仍能启动，并在工具调用时给出明确提示
    psutil = None

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Monitor_MCP_Server")

mcp = FastMCP("Monitor")


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


# ============================================================
# 辅助函数
# ============================================================

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
    # 返回默认时间（当前时间 + 偏移）
    return datetime.now() + timedelta(hours=default_offset_hours)


def generate_time_series(base_time: datetime, minutes_offset: int, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """生成时间序列字符串。

    Args:
        base_time: 基准时间
        minutes_offset: 分钟偏移量
        format_str: 时间格式字符串

    Returns:
        str: 格式化的时间字符串
    """
    result_time = base_time + timedelta(minutes=minutes_offset)
    return result_time.strftime(format_str)


def bytes_to_gb(value: int | float) -> float:
    """将字节转换为 GB。"""
    return round(float(value) / (1024 ** 3), 2)


def local_host_info() -> Dict[str, Any]:
    """返回当前 MCP Server 所在机器的基础信息。"""
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "system": platform.system(),
    }


def psutil_missing_response(tool_name: str) -> Dict[str, Any]:
    """psutil 未安装时的统一返回。"""
    return {
        "success": False,
        "metric_source": "local_machine",
        "tool": tool_name,
        "error": "psutil is not installed",
        "message": "本机指标采集依赖 psutil，请先安装依赖：pip install psutil，或重新安装项目依赖。",
    }


def clamp_sample_seconds(sample_seconds: float) -> float:
    """限制采样时长，避免工具调用被长时间阻塞。"""
    return max(0.1, min(float(sample_seconds), 5.0))


def default_disk_path() -> str:
    """根据操作系统选择默认磁盘检测路径。"""
    if os.name == "nt":
        return os.environ.get("SystemDrive", "C:") + "\\"
    return "/"




# ============================================================
# 监控数据查询工具
# ============================================================

@mcp.tool()
@log_tool_call
def query_cpu_metrics(
    service_name: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    interval: str = "1m",
    sample_seconds: float = 1.0,
) -> Dict[str, Any]:
    """查询当前 MCP Server 所在机器的实时 CPU 使用率。

    Args:
        service_name: 服务名称（兼容旧参数）。当前实现检测的是本机整体 CPU，
            service_name 仅作为诊断标签返回，不会按服务过滤。
        
        start_time: 开始时间（可选，字符串类型）
            格式: "YYYY-MM-DD HH:MM:SS"
            示例: "2026-02-14 10:00:00"
            默认值: 如果不传，默认为当前时间的1小时前
            注意: 必须使用字符串格式，而非时间戳
        
        end_time: 结束时间（可选，字符串类型）
            格式: "YYYY-MM-DD HH:MM:SS"
            示例: "2026-02-14 11:00:00"
            默认值: 如果不传，默认为当前时间
            注意: 必须使用字符串格式，而非时间戳
        
        interval: 数据聚合间隔（兼容旧参数）。当前本机采集不维护历史时间序列，
            因此只返回当前快照。
        sample_seconds: CPU 采样秒数，默认 1 秒，最大 5 秒。

    Returns:
        Dict: CPU 监控数据
            - service_name: 服务名称
            - metric_name: 指标名称 (cpu_usage_percent)
            - metric_source: local_machine_psutil
            - data_points: 当前快照数据点
            - statistics: 统计信息
            * current: 当前 CPU 使用率
            * per_cpu: 每个逻辑核心的使用率
            * load_average: Linux/macOS 的负载信息，Windows 下为 None
            - alert_info: 根据当前 CPU 使用率是否超过 80% 给出告警判断
    """
    if psutil is None:
        return psutil_missing_response("query_cpu_metrics")

    now = datetime.now()
    sample_seconds = clamp_sample_seconds(sample_seconds)
    cpu_value = round(psutil.cpu_percent(interval=sample_seconds), 1)
    per_cpu = [round(v, 1) for v in psutil.cpu_percent(interval=None, percpu=True)]
    cpu_freq = psutil.cpu_freq()
    load_average = list(os.getloadavg()) if hasattr(os, "getloadavg") else None
    spike_detected = cpu_value > 80.0

    data_point = {
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "value": cpu_value,
        "hostname": socket.gethostname(),
    }

    return {
        "success": True,
        "service_name": service_name,
        "metric_name": "cpu_usage_percent",
        "metric_source": "local_machine_psutil",
        "scope": "local_machine",
        "host": local_host_info(),
        "query_range": {
            "start_time": start_time,
            "end_time": end_time,
            "interval": interval,
            "note": "当前实现返回本机实时快照，不维护历史时间序列。",
        },
        "data_points": [data_point],
        "statistics": {
            "current": cpu_value,
            "avg": cpu_value,
            "max": cpu_value,
            "min": cpu_value,
            "p95": cpu_value,
            "per_cpu": per_cpu,
            "logical_cores": psutil.cpu_count(logical=True),
            "physical_cores": psutil.cpu_count(logical=False),
            "frequency_mhz": {
                "current": round(cpu_freq.current, 2) if cpu_freq else None,
                "min": round(cpu_freq.min, 2) if cpu_freq else None,
                "max": round(cpu_freq.max, 2) if cpu_freq else None,
            },
            "load_average": load_average,
            "spike_detected": spike_detected,
        },
        "alert_info": {
            "triggered": spike_detected,
            "threshold": 80.0,
            "message": "本机 CPU 使用率超过 80% 阈值" if spike_detected else "本机 CPU 使用率正常",
        },
    }


@mcp.tool()
@log_tool_call
def query_memory_metrics(
    service_name: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    interval: str = "1m"
) -> Dict[str, Any]:
    """查询当前 MCP Server 所在机器的实时内存使用情况。

    Args:
        service_name: 服务名称（兼容旧参数）。当前实现检测的是本机整体内存，
            service_name 仅作为诊断标签返回，不会按服务过滤。
        
        start_time: 开始时间（可选，字符串类型）
            格式: "YYYY-MM-DD HH:MM:SS"
            示例: "2026-02-14 10:00:00"
            默认值: 如果不传，默认为当前时间的1小时前
            注意: 必须使用字符串格式，而非时间戳
        
        end_time: 结束时间（可选，字符串类型）
            格式: "YYYY-MM-DD HH:MM:SS"
            示例: "2026-02-14 11:00:00"
            默认值: 如果不传，默认为当前时间
            注意: 必须使用字符串格式，而非时间戳
        
        interval: 数据聚合间隔（兼容旧参数）。当前本机采集不维护历史时间序列，
            因此只返回当前快照。

    Returns:
        Dict: 内存监控数据
            - service_name: 服务名称
            - metric_name: 指标名称 (memory_usage_percent)
            - metric_source: local_machine_psutil
            - data_points: 当前快照数据点
            - statistics: 统计信息
            * current: 当前内存使用率
            * total_gb / used_gb / available_gb: 内存容量
            * swap: 交换分区使用情况
            - alert_info: 根据当前内存使用率是否超过 70% 给出告警判断
    """
    if psutil is None:
        return psutil_missing_response("query_memory_metrics")

    now = datetime.now()
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    memory_value = round(memory.percent, 1)
    memory_pressure = memory_value > 70.0

    data_point = {
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "value": memory_value,
        "used_gb": bytes_to_gb(memory.used),
        "available_gb": bytes_to_gb(memory.available),
        "total_gb": bytes_to_gb(memory.total),
        "hostname": socket.gethostname(),
    }

    return {
        "success": True,
        "service_name": service_name,
        "metric_name": "memory_usage_percent",
        "metric_source": "local_machine_psutil",
        "scope": "local_machine",
        "host": local_host_info(),
        "query_range": {
            "start_time": start_time,
            "end_time": end_time,
            "interval": interval,
            "note": "当前实现返回本机实时快照，不维护历史时间序列。",
        },
        "data_points": [data_point],
        "statistics": {
            "current": memory_value,
            "avg": memory_value,
            "max": memory_value,
            "min": memory_value,
            "p95": memory_value,
            "total_gb": bytes_to_gb(memory.total),
            "used_gb": bytes_to_gb(memory.used),
            "available_gb": bytes_to_gb(memory.available),
            "free_gb": bytes_to_gb(memory.free),
            "swap": {
                "total_gb": bytes_to_gb(swap.total),
                "used_gb": bytes_to_gb(swap.used),
                "free_gb": bytes_to_gb(swap.free),
                "percent": round(swap.percent, 1),
            },
            "memory_pressure": memory_pressure,
        },
        "alert_info": {
            "triggered": memory_pressure,
            "threshold": 70.0,
            "message": "本机内存使用率超过 70% 阈值，存在内存压力" if memory_pressure else "本机内存使用率正常",
        },
    }


@mcp.tool()
@log_tool_call
def query_disk_metrics(path: Optional[str] = None) -> Dict[str, Any]:
    """查询当前 MCP Server 所在机器的磁盘使用情况。

    Args:
        path: 要检查的挂载点或磁盘路径。Windows 默认 C:\\，Linux/macOS 默认 /。

    Returns:
        Dict: 磁盘使用率、容量、分区列表和告警判断。
    """
    if psutil is None:
        return psutil_missing_response("query_disk_metrics")

    target_path = path or default_disk_path()
    try:
        usage = psutil.disk_usage(target_path)
    except Exception as e:
        return {
            "success": False,
            "metric_source": "local_machine_psutil",
            "scope": "local_machine",
            "host": local_host_info(),
            "path": target_path,
            "error": str(e),
            "message": f"磁盘路径不可用: {target_path}",
        }

    partitions = []
    for part in psutil.disk_partitions(all=False):
        try:
            part_usage = psutil.disk_usage(part.mountpoint)
        except Exception:
            continue
        partitions.append(
            {
                "device": part.device,
                "mountpoint": part.mountpoint,
                "fstype": part.fstype,
                "total_gb": bytes_to_gb(part_usage.total),
                "used_gb": bytes_to_gb(part_usage.used),
                "free_gb": bytes_to_gb(part_usage.free),
                "percent": round(part_usage.percent, 1),
            }
        )

    disk_pressure = usage.percent > 85.0
    return {
        "success": True,
        "metric_name": "disk_usage_percent",
        "metric_source": "local_machine_psutil",
        "scope": "local_machine",
        "host": local_host_info(),
        "path": target_path,
        "statistics": {
            "total_gb": bytes_to_gb(usage.total),
            "used_gb": bytes_to_gb(usage.used),
            "free_gb": bytes_to_gb(usage.free),
            "percent": round(usage.percent, 1),
        },
        "partitions": partitions,
        "alert_info": {
            "triggered": disk_pressure,
            "threshold": 85.0,
            "message": "本机磁盘使用率超过 85% 阈值" if disk_pressure else "本机磁盘使用率正常",
        },
    }


@mcp.tool()
@log_tool_call
def query_process_list(limit: int = 10, sort_by: str = "memory") -> Dict[str, Any]:
    """查询当前 MCP Server 所在机器上 CPU 或内存占用最高的进程。

    Args:
        limit: 返回进程数量，默认 10，最大 50。
        sort_by: 排序方式，支持 memory 或 cpu。

    Returns:
        Dict: 进程列表，包含 pid、名称、用户、CPU%、内存%、RSS 内存等。
    """
    if psutil is None:
        return psutil_missing_response("query_process_list")

    limit = max(1, min(int(limit), 50))
    sort_by = sort_by.lower()
    if sort_by not in ("memory", "cpu"):
        sort_by = "memory"

    processes = []
    raw_processes = list(psutil.process_iter(["pid", "name", "username", "status", "create_time"]))

    for proc in raw_processes:
        try:
            proc.cpu_percent(interval=None)
        except Exception:
            continue

    time.sleep(0.2)

    for proc in raw_processes:
        try:
            info = proc.info
            memory_info = proc.memory_info()
            create_time = datetime.fromtimestamp(info.get("create_time", 0)).strftime("%Y-%m-%d %H:%M:%S")
            processes.append(
                {
                    "pid": info.get("pid"),
                    "name": info.get("name"),
                    "username": info.get("username"),
                    "status": info.get("status"),
                    "cpu_percent": round(proc.cpu_percent(interval=None), 1),
                    "memory_percent": round(proc.memory_percent(), 2),
                    "rss_gb": bytes_to_gb(memory_info.rss),
                    "create_time": create_time,
                }
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    sort_key = "cpu_percent" if sort_by == "cpu" else "memory_percent"
    processes.sort(key=lambda item: item.get(sort_key, 0), reverse=True)

    return {
        "success": True,
        "metric_source": "local_machine_psutil",
        "scope": "local_machine",
        "host": local_host_info(),
        "sort_by": sort_by,
        "total_seen": len(processes),
        "processes": processes[:limit],
        "message": f"已获取本机 {len(processes[:limit])} 个进程，按 {sort_by} 占用排序",
    }


@mcp.tool()
@log_tool_call
def get_local_system_info() -> Dict[str, Any]:
    """查询当前 MCP Server 所在机器的基础系统信息。"""
    if psutil is None:
        return psutil_missing_response("get_local_system_info")

    boot_time = datetime.fromtimestamp(psutil.boot_time())
    uptime = datetime.now() - boot_time
    memory = psutil.virtual_memory()

    return {
        "success": True,
        "metric_source": "local_machine_psutil",
        "scope": "local_machine",
        "host": local_host_info(),
        "boot_time": boot_time.strftime("%Y-%m-%d %H:%M:%S"),
        "uptime_seconds": int(uptime.total_seconds()),
        "cpu": {
            "logical_cores": psutil.cpu_count(logical=True),
            "physical_cores": psutil.cpu_count(logical=False),
        },
        "memory": {
            "total_gb": bytes_to_gb(memory.total),
            "available_gb": bytes_to_gb(memory.available),
            "percent": round(memory.percent, 1),
        },
        "disk_default_path": default_disk_path(),
    }




if __name__ == "__main__":
    # 使用 streamable-http 模式，运行在 8004 端口
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8004, path="/mcp")
