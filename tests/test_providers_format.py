from traject.models import Message, ToolCall, ToolSpec
from traject.providers.anthropic_ import to_anthropic_messages, to_anthropic_tools
from traject.providers.openai_ import to_openai_messages, to_openai_tools

HIST = [
    Message(role="user", content="hi"),
    Message(role="assistant", content="", tool_calls=[
        ToolCall(id="c1", name="a", input={"x": 1}),
        ToolCall(id="c2", name="b", input={"y": 2}),
    ]),
    Message(role="tool", tool_call_id="c1", name="a", content="ra"),
    Message(role="tool", tool_call_id="c2", name="b", content="rb"),
    Message(role="assistant", content="done"),
]


def test_anthropic_groups_parallel_tool_results_into_one_user_message():
    out = to_anthropic_messages(HIST)
    assert [m["role"] for m in out] == ["user", "assistant", "user", "assistant"]
    results = out[2]["content"]
    assert [b["tool_use_id"] for b in results] == ["c1", "c2"]
    assert all(b["type"] == "tool_result" for b in results)
    assert out[1]["content"][0]["type"] == "tool_use"  # no empty text block before tool_use


def test_openai_tool_messages_and_arguments_are_json_strings():
    out = to_openai_messages("sys", HIST)
    assert out[0] == {"role": "system", "content": "sys"}
    asst = out[2]
    assert asst["content"] is None
    assert asst["tool_calls"][0]["function"]["arguments"] == '{"x": 1}'
    assert out[3] == {"role": "tool", "tool_call_id": "c1", "content": "ra"}


def test_tool_spec_conversion():
    spec = ToolSpec(name="t", description="d", input_schema={"type": "object", "properties": {}})
    assert to_anthropic_tools([spec])[0]["input_schema"] == spec.input_schema
    assert to_openai_tools([spec])[0]["function"]["parameters"] == spec.input_schema
