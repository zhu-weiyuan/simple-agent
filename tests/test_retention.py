from my_agent.tools.retention import ItemRetainer, format_text_notice, retain_text


def test_item_retainer_counts_omitted_items():
    retainer = ItemRetainer(2)
    retainer.extend(["a", "b", "c", "d"])
    result = retainer.finish()
    assert result.items == ["a", "b"]
    assert result.omitted == 2
    assert result.truncated
    assert "省略 2 项" in result.notice("搜索结果")


def test_retain_text_reports_exact_omission():
    value, omitted = retain_text("abcdef", 4)
    assert value == "abcd"
    assert omitted == 2
    assert "省略 2 个字符" in format_text_notice("输出", omitted)


def test_retain_text_rejects_invalid_budget():
    try:
        retain_text("x", 0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
