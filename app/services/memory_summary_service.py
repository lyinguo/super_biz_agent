"""会话记忆压缩服务.

按 session_id 独立维护长期摘要。当某个会话的历史消息超过阈值时，
将较早的消息压缩为 Markdown 摘要，并只保留摘要 + 最近几轮真实消息。
"""

import asyncio
from pathlib import Path
import re
from typing import Any, Hashable

from langchain_core.messages import BaseMessage, RemoveMessage, SystemMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from loguru import logger

from app.config import config
from app.core.llm_factory import llm_factory


class MemorySummaryService:
    """按会话压缩 LangGraph checkpointer 中的历史消息."""

    def __init__(
        self,
        summary_dir: str | None = None,
        max_chars: int | None = None,
        keep_recent: int | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.enabled = config.memory_summary_enabled if enabled is None else enabled
        self.summary_dir = Path(summary_dir or config.memory_summary_dir)
        self.summary_dir.mkdir(parents=True, exist_ok=True)
        self.max_chars = max_chars or config.memory_summary_max_chars
        self.keep_recent = keep_recent or config.memory_summary_keep_recent
        self._running_sessions: set[str] = set()

    def _safe_session_id(self, session_id: str) -> str:
        """将 session_id 转成可作为文件名使用的字符串."""
        return re.sub(r"[^a-zA-Z0-9_.-]", "_", session_id)

    def _summary_path(self, session_id: str) -> Path:
        return self.summary_dir / f"{self._safe_session_id(session_id)}.md"

    def load_summary(self, session_id: str) -> str:
        """读取指定会话的长期摘要."""
        path = self._summary_path(session_id)
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def save_summary(self, session_id: str, summary: str) -> None:
        """保存指定会话的长期摘要."""
        self.summary_dir.mkdir(parents=True, exist_ok=True)
        self._summary_path(session_id).write_text(summary, encoding="utf-8")

    def delete_summary(self, session_id: str) -> None:
        """删除指定会话的摘要文件."""
        path = self._summary_path(session_id)
        if path.exists():
            path.unlink()

    def _message_text(self, msg: BaseMessage) -> str:
        """将 LangChain message 转成适合总结模型阅读的文本."""
        role = getattr(msg, "type", type(msg).__name__)
        content = getattr(msg, "content", "")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text") or item.get("content") or item))
                else:
                    parts.append(str(item))
            content = "\n".join(parts)
        return f"{role}: {content}"

    def _total_chars(self, messages: list[BaseMessage]) -> int:
        return sum(len(self._message_text(m)) for m in messages)

    def should_summarize(self, messages: list[BaseMessage]) -> bool:
        """判断当前历史是否需要压缩."""
        if not self.enabled:
            return False
        if len(messages) <= self.keep_recent + 4:
            return False
        return self._total_chars(messages) > self.max_chars

    async def summarize(
        self,
        session_id: str,
        old_messages: list[BaseMessage],
        save: bool = True,
    ) -> str:
        """调用总结模型，将旧历史压缩成长期摘要."""
        previous_summary = self.load_summary(session_id)
        history_text = "\n\n".join(self._message_text(m) for m in old_messages)

        prompt = f"""
                你是一个会话记忆总结助手。请把历史对话压缩成长期记忆摘要。

                要求：
                1. 保留用户长期目标、偏好、已经达成的结论
                2. 保留重要技术决策、排查结果、代码改动方向
                3. 保留后续对话可能需要继续承接的待办事项
                4. 删除寒暄、重复内容、无关细节
                5. 使用 Markdown 输出，分成「会话摘要」「关键结论」「用户偏好」「待办/后续关注」
                6. 不要编造历史中没有的信息

                已有摘要：
                {previous_summary or "无"}

                本次需要压缩的历史：
                {history_text}
                """.strip()

        llm = llm_factory.create_chat_model(
            temperature=0,
            streaming=False,
        )
        response = await llm.ainvoke(prompt)
        summary = response.content if hasattr(response, "content") else str(response)
        summary = summary.strip()

        if save:
            self.save_summary(session_id, summary)
        return summary

    def _extract_messages_from_state(self, state: Any) -> list[BaseMessage]:
        """兼容 LangGraph StateSnapshot / dict 两类返回结构."""
        values = getattr(state, "values", None)
        if values is None and isinstance(state, dict):
            values = state.get("values", state)
        values = values or {}
        messages = values.get("messages", [])
        return list(messages or [])

    async def _get_agent_messages(
        self,
        agent: Any,
        state_config: dict[str, Any],
    ) -> list[BaseMessage]:
        """读取指定 thread 的当前 messages."""
        if hasattr(agent, "aget_state"):
            state = await agent.aget_state(state_config)
        else:
            state = agent.get_state(state_config)
        return self._extract_messages_from_state(state)

    def _message_signature(self, messages: list[BaseMessage]) -> tuple[Hashable, ...]:
        """生成轻量签名，用于判断后台总结期间会话是否发生变化."""
        signature = []
        for msg in messages:
            text = self._message_text(msg)
            signature.append(
                (
                    getattr(msg, "id", None),
                    getattr(msg, "type", type(msg).__name__),
                    len(text),
                    text[:120],
                )
            )
        return tuple(signature)

    def _build_compressed_messages(
        self,
        summary: str,
        recent_messages: list[BaseMessage],
    ) -> list[BaseMessage]:
        """构建用于替换 LangGraph messages 的压缩后消息列表."""
        return [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            SystemMessage(content=f"以下是本会话较早历史的长期摘要，请在后续回答中参考：\n\n{summary}"),
            *recent_messages,
        ]

    async def maybe_compress_agent_state(
        self,
        agent: Any,
        session_id: str,
    ) -> bool:
        """同步压缩：如果指定 session 的历史过长，则立即压缩 LangGraph 状态."""
        return await self._compress_agent_state(
            agent=agent,
            session_id=session_id,
            verify_unchanged=False,
            source="sync",
        )

    def schedule_compress_agent_state(
        self,
        agent: Any,
        session_id: str,
    ) -> bool:
        """异步压缩：调度后台任务，不阻塞当前对话请求."""
        if not self.enabled or agent is None:
            return False

        if session_id in self._running_sessions:
            logger.debug("[会话 {}] 已有异步历史压缩任务在运行，跳过重复调度", session_id)
            return False

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("[会话 {}] 当前没有运行中的事件循环，无法调度异步历史压缩", session_id)
            return False

        self._running_sessions.add(session_id)
        task = loop.create_task(self._run_background_compression(agent, session_id))
        task.add_done_callback(lambda t: self._on_background_done(session_id, t))
        logger.info("[会话 {}] 已调度异步历史压缩任务", session_id)
        return True

    async def _run_background_compression(self, agent: Any, session_id: str) -> bool:
        """后台执行历史压缩；更新前会校验会话未被新消息改变."""
        return await self._compress_agent_state(
            agent=agent,
            session_id=session_id,
            verify_unchanged=True,
            source="async",
        )

    def _on_background_done(self, session_id: str, task: asyncio.Task) -> None:
        """清理后台任务状态并记录异常."""
        self._running_sessions.discard(session_id)
        try:
            _ = task.result()
        except Exception as e:
            logger.warning("[会话 {}] 异步历史压缩任务异常: {}", session_id, e)

    async def _compress_agent_state(
        self,
        agent: Any,
        session_id: str,
        verify_unchanged: bool,
        source: str,
    ) -> bool:
        """执行压缩；source 用于区分 sync / async 日志."""
        if not self.enabled or agent is None:
            return False

        state_config = {"configurable": {"thread_id": session_id}}

        try:
            messages = await self._get_agent_messages(agent, state_config)
            if not messages or not self.should_summarize(messages):
                return False

            initial_signature = self._message_signature(messages)

            non_system_messages = [
                m for m in messages
                if getattr(m, "type", "") != "system"
            ]

            old_messages = non_system_messages[:-self.keep_recent]
            recent_messages = non_system_messages[-self.keep_recent:]
            if not old_messages:
                return False

            logger.info(
                "[会话 {}] 历史消息过长，开始{}压缩: total_messages={}, old={}, recent={}",
                session_id,
                "异步" if source == "async" else "同步",
                len(messages),
                len(old_messages),
                len(recent_messages),
            )

            summary = await self.summarize(session_id, old_messages, save=False)

            if verify_unchanged:
                latest_messages = await self._get_agent_messages(agent, state_config)
                if self._message_signature(latest_messages) != initial_signature:
                    logger.info(
                        "[会话 {}] 异步压缩期间会话已有新消息，跳过本次状态替换",
                        session_id,
                    )
                    return False

            compressed_messages = self._build_compressed_messages(summary, recent_messages)
            update = {"messages": compressed_messages}

            if hasattr(agent, "aupdate_state"):
                await agent.aupdate_state(state_config, update)
            else:
                agent.update_state(state_config, update)

            self.save_summary(session_id, summary)

            logger.info(
                "[会话 {}] {}历史压缩完成，保留 recent_messages={}，摘要长度={}",
                session_id,
                "异步" if source == "async" else "同步",
                len(recent_messages),
                len(summary),
            )
            return True

        except Exception as e:
            logger.warning("[会话 {}] {}历史压缩失败，将继续正常对话: {}", session_id, source, e)
            return False


memory_summary_service = MemorySummaryService()
