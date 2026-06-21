from cscall.streaming.local_agreement import LocalAgreement


def test_commits_only_after_two_matching_prefixes():
    stabilizer = LocalAgreement()

    first = stabilizer.update("Hello wor")
    second = stabilizer.update("hello wor!")

    assert first.committed == ""
    assert first.unstable == "hello wor"
    assert second.committed == "hello wor"
    assert second.unstable == ""


def test_changed_suffix_stays_unstable_until_it_repeats():
    stabilizer = LocalAgreement()

    stabilizer.update("hello wor")
    stabilizer.update("hello wor!")

    third = stabilizer.update("hello world")
    fourth = stabilizer.update("hello world")

    assert third.committed == ""
    assert third.unstable == "ld"
    assert fourth.committed == "ld"
    assert fourth.unstable == ""


def test_final_flush_emits_remaining_uncommitted_text():
    stabilizer = LocalAgreement()

    stabilizer.update("keep this")

    assert stabilizer.final_flush() == "keep this"
    assert stabilizer.final_flush() == ""


def test_empty_hypotheses_do_not_crash_or_commit():
    stabilizer = LocalAgreement()

    first = stabilizer.update("")
    second = stabilizer.update(" ")
    third = stabilizer.update("hello")
    fourth = stabilizer.update("")

    assert first.committed == ""
    assert first.unstable == ""
    assert second.committed == ""
    assert second.unstable == ""
    assert third.committed == ""
    assert third.unstable == "hello"
    assert fourth.committed == ""
    assert fourth.unstable == "hello"


def test_punctuation_and_whitespace_differences_still_match():
    stabilizer = LocalAgreement(agreement=2)

    first = stabilizer.update("Hello,   world")
    second = stabilizer.update("hello world!!")

    assert first.committed == ""
    assert first.unstable == "hello world"
    assert second.committed == "hello world"
    assert second.unstable == ""
