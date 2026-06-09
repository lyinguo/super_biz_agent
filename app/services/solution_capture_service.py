"""对话解决方案沉淀服务.

在对话完成后异步判断本轮问答是否包含可复用解决方案。
如果值得沉淀，则生成结构化 Markdown 文件，并可选自动写入向量知识库。
"""

import asyncio
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from app.config import config
from app.core.llm_factory import llm_factory


class CapturedSolution(BaseModel):
    """解决方案沉淀结构."""

    should_capture: bool = Field(description="是否值得沉淀为可复用解决方案")
    title: str = Field(default="", description="简短标题，适合做 Markdown 文件标题")
    problem: str = Field(default="", description="问题/场景描述")
    context: str = Field(default="", description="适用背景和前置条件")
    root_cause: str = Field(default="", description="原因分析；如果无法确定则说明未知")
    solution_steps: list[str] = Field(default_factory=list, description="可执行处理步骤")
    verification: list[str] = Field(default_factory=list, description="验证方式")
    prevention: list[str] = Field(default_factory=list, description="预防或长期优化建议")
    tags: list[str] = Field(default_factory=list, description="检索标签")


class SolutionCaptureService:
    """把高价值问答沉淀为共享知识文档."""

    def __init__(self) -> None:
        self.enabled = config.solution_capture_enabled
        self.solution_dir = Path(config.solution_capture_dir)
        self.auto_index = config.solution_capture_auto_index
        self.min_chars = config.solution_capture_min_chars
        self.solution_dir.mkdir(parents=True, exist_ok=True)

    def _looks_like_solution(self, question: str, answer: str) -> bool:
        """先用低成本规则过滤明显不值得沉淀的对话."""
        combined = f"{question}\n{answer}"
        if len(combined.strip()) < self.min_chars:
            return False

        keywords = (
            "解决", "修复", "排查", "原因", "步骤", "方案", "配置", "报错",
            "启动", "部署", "接口", "数据库", "docker", "milvus", "mcp",
            "agent", "异常", "优化", "怎么做", "怎么办", "为什么",
        )
        return any(k.lower() in combined.lower() for k in keywords)

    def _safe_filename_part(self, text: str) -> str:
        """从文本生成安全的文件名部分."""
        text = re.sub(r"[\\/:*?\"<>|]", "_", text)
        text = re.sub(r"\s+", "_", text.strip())
        text = re.sub(r"_+", "_", text)
        return text[:60].strip("_") or "solution"

    def _build_markdown(
        self,
        session_id: str,
        question: str,
        solution: CapturedSolution,
    ) -> str:
        tags = ", ".join(solution.tags) if solution.tags else "未分类"
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        def list_block(items: list[str]) -> str:
            if not items:
                return "- 暂无"
            return "\n".join(f"{i + 1}. {item}" for i, item in enumerate(items))

        return f"""# {solution.title or "可复用解决方案"}

## 元信息
- **创建时间**: {created_at}
- **来源会话**: {session_id}
- **标签**: {tags}

## 原始问题
{question.strip()}

## 适用场景
{solution.context or "暂无明确适用场景。"}

## 问题描述
{solution.problem or "暂无问题描述。"}

## 原因分析
{solution.root_cause or "未能从对话中确定根因。"}

## 处理步骤
{list_block(solution.solution_steps)}

## 验证方式
{list_block(solution.verification)}

## 预防和长期优化
{list_block(solution.prevention)}

## 复用提示
- 后续遇到类似问题时，可先确认适用场景是否一致。
- 如果环境、版本或配置不同，需要结合现场信息调整步骤。
"""

    async def _extract_solution(self, question: str, answer: str) -> CapturedSolution | None:
        """调用 LLM 判断并抽取可复用方案."""
        prompt = f"""
你是一个企业知识库沉淀助手。请判断下面这轮问答是否包含值得沉淀的可复用解决方案。

沉淀标准：
1. 解决了具体问题，或给出了可执行排查/修复/配置/优化步骤
2. 对其他用户或新人以后处理类似问题有帮助
3. 不沉淀普通寒暄、纯概念闲聊、没有明确步骤的回答
4. 去掉个人化表达，保留可复用知识

请输出结构化结果。

用户问题：
{question}

助手回答：
{answer}
""".strip()

        llm = llm_factory.create_chat_model(temperature=0, streaming=False)
        try:
            structured_llm = llm.with_structured_output(CapturedSolution)
            result = await structured_llm.ainvoke(prompt)
            if isinstance(result, CapturedSolution):
                return result
            if isinstance(result, dict):
                return CapturedSolution(**result)
        except Exception as e:
            logger.warning("结构化方案抽取失败，尝试 JSON 兜底解析: {}", e)

        json_prompt = f"""{prompt}

请只输出 JSON，不要输出 Markdown。JSON 字段如下：
{{
  "should_capture": true,
  "title": "",
  "problem": "",
  "context": "",
  "root_cause": "",
  "solution_steps": [],
  "verification": [],
  "prevention": [],
  "tags": []
}}
""".strip()
        response = await llm.ainvoke(json_prompt)
        content = response.content if hasattr(response, "content") else str(response)
        content = str(content).strip()
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            return None
        return CapturedSolution(**json.loads(match.group(0)))

    async def capture_if_useful(
        self,
        session_id: str,
        question: str,
        answer: str,
    ) -> Path | None:
        """如果本轮问答值得沉淀，则生成 md 并可选写入向量库."""
        if not self.enabled:
            return None
        if not self._looks_like_solution(question, answer):
            return None

        try:
            solution = await self._extract_solution(question, answer)
            if not solution or not solution.should_capture:
                logger.debug("[会话 {}] 本轮问答不需要沉淀为解决方案", session_id)
                return None

            title_part = self._safe_filename_part(solution.title or question)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = self.solution_dir / f"{timestamp}_{title_part}.md"
            markdown = self._build_markdown(session_id, question, solution)
            file_path.write_text(markdown, encoding="utf-8")

            logger.info("[会话 {}] 已沉淀解决方案: {}", session_id, file_path)

            if self.auto_index:
                await self._index_solution_file(file_path)

            return file_path

        except Exception as e:
            logger.warning("[会话 {}] 解决方案沉淀失败: {}", session_id, e)
            return None

    async def _index_solution_file(self, file_path: Path) -> None:
        """将生成的解决方案写入向量知识库."""
        try:
            from app.services.vector_index_service import vector_index_service

            await asyncio.to_thread(vector_index_service.index_single_file, str(file_path))
            logger.info("解决方案已写入向量知识库: {}", file_path)
        except Exception as e:
            logger.warning("解决方案写入向量知识库失败: {}, 错误: {}", file_path, e)

    def schedule_capture(
        self,
        session_id: str,
        question: str,
        answer: str,
    ) -> bool:
        """后台调度，不阻塞当前对话."""
        if not self.enabled:
            return False
        if not self._looks_like_solution(question, answer):
            return False

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("[会话 {}] 当前没有运行中的事件循环，无法调度解决方案沉淀", session_id)
            return False

        task = loop.create_task(self.capture_if_useful(session_id, question, answer))
        task.add_done_callback(lambda t: self._on_capture_done(session_id, t))
        logger.info("[会话 {}] 已调度解决方案沉淀任务", session_id)
        return True

    def _on_capture_done(self, session_id: str, task: asyncio.Task) -> None:
        try:
            _ = task.result()
        except Exception as e:
            logger.warning("[会话 {}] 解决方案沉淀后台任务异常: {}", session_id, e)


solution_capture_service = SolutionCaptureService()
