from conversation import ConversationManager


def test_latest_update_follow_up_is_contextualized():
    manager = ConversationManager()
    state = manager.set_history(
        "test-1",
        [{"role": "user", "content": "What are the latest Power BI updates?"}],
    )

    plan = manager.contextualize("Is there any recent update regarding RLS?", state)

    assert plan["is_follow_up"] is True
    assert "latest Power BI updates" in plan["query"]
    assert "RLS" in plan["query"]


def test_ols_follow_up_keeps_release_context():
    manager = ConversationManager()
    state = manager.set_history(
        "test-2",
        [{"role": "user", "content": "What are the latest Power BI updates?"}],
    )

    plan = manager.contextualize("What is OLS?", state)

    assert plan["is_follow_up"] is True
    assert "Power BI" in plan["query"]
    assert "OLS" in plan["query"]


def test_standalone_question_does_not_force_context():
    manager = ConversationManager()
    state = manager.set_history("test-3", [])

    plan = manager.contextualize("Explain CALCULATE in DAX", state)

    assert plan["is_follow_up"] is False
    assert plan["query"] == "Explain CALCULATE in DAX"
