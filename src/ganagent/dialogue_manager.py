from __future__ import annotations

from dataclasses import dataclass
import re

from ganagent.models import AgentResult
from ganagent.product import TranslationProduct


@dataclass(frozen=True)
class DialogueReply:
    text: str
    source: str
    needs_codex_search: bool = False


def build_dialogue_reply(product: TranslationProduct, result: AgentResult) -> DialogueReply:
    """Build a conservative spoken answer for one live dialogue turn."""

    if product.status == "unreliable":
        return DialogueReply(
            text="这句话我没有听清楚。请靠近麦克风，再用短一点的一句话讲一遍。",
            source="risk_fallback",
        )
    question = _clean_text(product.mandarin or result.transcript)
    if not question:
        return DialogueReply(
            text="我这边没有听到清楚的问题。请再说一遍。",
            source="empty_fallback",
        )

    local_reply = _answer_common_service_question(question)
    if local_reply:
        return DialogueReply(text=local_reply, source="local_service_rules")

    if _looks_like_search_question(question):
        return DialogueReply(
            text=(
                "这个问题可能需要查最新信息。我已经把识别结果整理成 Codex 联网任务，"
                "请在 Codex 里继续搜索核对后回答。"
            ),
            source="codex_search_task",
            needs_codex_search=True,
        )

    return DialogueReply(
        text=f"我听到你的意思是：{question}。如果你是在问办事或求助流程，可以再补充一个具体问题。",
        source="echo_clarify",
    )


def _answer_common_service_question(question: str) -> str | None:
    if any(term in question for term in ("身份证", "证件", "居民证")) and any(
        term in question for term in ("丢", "掉", "遗失", "不见", "补")
    ):
        return (
            "身份证遗失的话，建议先到就近派出所或户籍窗口咨询补办。"
            "如果不确定材料，可以先拨打市民服务热线一二三四五确认。"
        )
    if "居住证" in question and any(term in question for term in ("办理", "办", "申请", "补办")):
        return (
            "居住证办理通常要准备身份证明、居住地址证明等材料。"
            "不同社区要求可能不一样，建议先问居委会或拨打一二三四五确认。"
        )
    if re.search(r"110|报警|警察", question):
        return "遇到人身危险、盗抢或紧急治安问题，请立即拨打一一零报警。"
    if re.search(r"119|火警|着火|消防", question):
        return "遇到火灾或明显消防危险，请立即拨打一一九，并尽快撤离到安全位置。"
    if re.search(r"120|急救|救护|医院", question):
        return "遇到突发疾病或严重受伤，请立即拨打一二零急救电话。"
    if re.search(r"12345|热线|投诉|咨询", question):
        return "一般政务咨询、投诉或求助，可以拨打一二三四五市民服务热线。"
    if any(term in question for term in ("你好", "侬好", "喂", "在吗")):
        return "侬好，我在。你可以直接用上海话问我办事、求助或生活问题。"
    return None


def _looks_like_search_question(question: str) -> bool:
    return any(
        term in question
        for term in (
            "今天",
            "现在",
            "最新",
            "新闻",
            "天气",
            "价格",
            "营业",
            "地址",
            "电话",
            "政策",
            "规定",
            "怎么去",
            "几点",
        )
    )


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", "", text or "").strip()
