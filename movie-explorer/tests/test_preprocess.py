from utils.preprocess import clean_genres, code_to_language


def test_clean_genres_splits_and_sorts():
    assert clean_genres("Drama, Action, Drama") == ["Action", "Drama"]


def test_clean_genres_handles_missing_value():
    assert clean_genres(None) == []


def test_code_to_language_converts_iso_code():
    assert code_to_language("en") == "English"


def test_code_to_language_handles_chinese_alias():
    assert code_to_language("cn") == "Chinese"


def test_code_to_language_handles_unknown_code():
    assert code_to_language("xx") == "XX"
