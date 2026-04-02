# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ast, json, re, uuid
from collections.abc import Sequence
from typing import Union

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger
from vllm.tool_parsers.abstract_tool_parser import ToolParser, ToolParserManager
from vllm.tokenizers import TokenizerLike

logger = init_logger(__name__)

_TOOL_CALL_BLOCK_RE = re.compile(
    r"<\|tool_call_start\|>\[(.+?)\]<\|tool_call_end\|>", re.DOTALL
)
_TOOL_CALL_START_TOKEN = "<|tool_call_start|>"
_TOOL_CALL_END_TOKEN = "<|tool_call_end|>"
_FUNC_CALL_RE = re.compile(r"([a-zA-Z_]\w*)\s*\(", re.DOTALL)


def _parse_pythonic_args(args_str: str) -> dict:
    try:
        tree = ast.parse(f"_f({args_str})", mode="eval")
        return {kw.arg: ast.literal_eval(kw.value) for kw in tree.body.keywords}
    except Exception:
        return _regex_parse_args(args_str)


def _regex_parse_args(args_str: str) -> dict:
    result = {}
    pattern = re.compile(
        r'(\w+)\s*=\s*("(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|True|False|None|-?\d+\.\d+|-?\d+)'
    )
    for m in pattern.finditer(args_str):
        try:
            result[m.group(1)] = ast.literal_eval(m.group(2))
        except Exception:
            result[m.group(1)] = m.group(2).strip("\"'")
    return result


def _split_top_level_calls(inner: str) -> list:
    calls, depth, buf = [], 0, []
    for ch in inner:
        if ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth -= 1
            buf.append(ch)
            if depth == 0:
                calls.append("".join(buf).strip())
                buf = []
        elif ch == "," and depth == 0:
            pass
        else:
            buf.append(ch)
    if leftover := "".join(buf).strip():
        calls.append(leftover)
    return [c for c in calls if c]


def _parse_tool_calls(inner: str) -> list:
    tool_calls = []
    for raw in _split_top_level_calls(inner):
        m = _FUNC_CALL_RE.match(raw.strip())
        if not m:
            continue
        func_name = m.group(1)
        paren_start = raw.index("(")
        args_str = raw[paren_start + 1 : raw.rindex(")")].strip()
        args_dict = _parse_pythonic_args(args_str) if args_str else {}
        tool_calls.append(
            ToolCall(
                id=make_tool_call_id(),
                type="function",
                function=FunctionCall(name=func_name, arguments=json.dumps(args_dict)),
            )
        )
    return tool_calls


@ToolParserManager.register_module(["lfm"])
class LFMToolParser(ToolParser):
    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)
        self._tool_call_buffer = ""
        self._streamed_args: dict = {}

    def extract_tool_calls(
        self, model_output: str, request
    ) -> ExtractedToolCallInformation:
        match = _TOOL_CALL_BLOCK_RE.search(model_output)
        if not match:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output or None
            )
        tool_calls = _parse_tool_calls(match.group(1))
        if not tool_calls:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output or None
            )
        content = _TOOL_CALL_BLOCK_RE.sub("", model_output).strip() or None
        return ExtractedToolCallInformation(
            tools_called=True, tool_calls=tool_calls, content=content
        )

    def extract_tool_calls_streaming(
        self,
        previous_text,
        current_text,
        delta_text,
        previous_token_ids,
        current_token_ids,
        delta_token_ids,
        request,
    ):
        self._tool_call_buffer = current_text
        if _TOOL_CALL_END_TOKEN in self._tool_call_buffer:
            match = _TOOL_CALL_BLOCK_RE.search(self._tool_call_buffer)
            if match:
                tool_calls = _parse_tool_calls(match.group(1))
                if not tool_calls:
                    return None
                delta_tool_calls = []
                for idx, tc in enumerate(tool_calls):
                    already_sent = self._streamed_args.get(idx, "")
                    new_args = tc.function.arguments[len(already_sent) :]
                    self._streamed_args[idx] = tc.function.arguments
                    delta_tool_calls.append(
                        DeltaToolCall(
                            index=idx,
                            type="function",
                            id=tc.id if not already_sent else None,
                            function=DeltaFunctionCall(
                                name=tc.function.name if not already_sent else None,
                                arguments=new_args,
                            ),
                        )
                    )
                content_before = (
                    _TOOL_CALL_BLOCK_RE.sub("", self._tool_call_buffer).strip() or None
                )
                return DeltaMessage(
                    content=content_before, tool_calls=delta_tool_calls or None
                )
        if _TOOL_CALL_START_TOKEN in self._tool_call_buffer:
            return DeltaMessage(content=None, tool_calls=None)
        return DeltaMessage(content=delta_text, tool_calls=None)
