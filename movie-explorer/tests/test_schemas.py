from chatbot.schemas import UserQuery


def test_semantic_text_strips_description():
    query = UserQuery(genres=["Action"], description="  sci-fi epic  ")
    assert query.semantic_text() == "sci-fi epic"


def test_semantic_text_empty_when_no_description():
    query = UserQuery(genres=["Drama"])
    assert query.semantic_text() == ""
