from dataclasses import dataclass
from typing import Any, Optional
import pytest

from src.solution_tree import Matcher, Query, SolutionTree, ValueMatcher


def vmatch(*values: Any) -> ValueMatcher[Any, Any]:
    return ValueMatcher(lambda x: x, values)


@pytest.mark.parametrize(
    ["left", "right", "expected"],
    [
        pytest.param(
            vmatch("a", "b", "c"),
            vmatch("a", "b", "c"),
            vmatch("a", "b", "c"),
            id="value matcher identity",
        ),
        pytest.param(
            vmatch("a", "c"),
            vmatch("a", "b", "c"),
            vmatch("a", "c"),
            id="value matcher intersection 1",
        ),
        pytest.param(
            vmatch("c"),
            vmatch("a", "b", "c"),
            vmatch("c"),
            id="value matcher intersection 2",
        ),
        pytest.param(
            vmatch("a", "b", "c"),
            vmatch("a", "c"),
            vmatch("a", "c"),
            id="value matcher intersection 3",
        ),
        pytest.param(
            vmatch(None, "a", "b"),
            vmatch("a", "b", "c"),
            vmatch("a", "b"),
            id="None is also a value",
        ),
        pytest.param(
            vmatch(None), vmatch("a", "b", "c"), None, id="Empty intersection 1"
        ),
        pytest.param(vmatch("b"), vmatch("a", "c"), None, id="Empty intersection 2"),
    ],
)
def test_value_matcher_intersection(
    left: Matcher[str | None],
    right: Matcher[str | None],
    expected: Optional[Matcher[str | None]],
) -> None:
    assert left.intersect(right) == expected


@pytest.mark.parametrize(
    ["value", "matcher", "expected"],
    [
        pytest.param("a", vmatch("a", "b", "c"), True),
        pytest.param("b", vmatch("a", "b", "c"), True),
        pytest.param("d", vmatch("a", "b", "c"), False),
        pytest.param(None, vmatch("a", "b", "c"), False),
        pytest.param(123, vmatch("a", "b", "c"), False),
    ],
)
def test_value_matcher(
    value: Any, matcher: Matcher[str | None], expected: bool
) -> None:
    assert matcher.match(value) == expected


temp_selectors = {
    "a": lambda _: 1,
    "b": lambda _: 2,
    "c": lambda _: 3,
}


@dataclass
class Apple:
    family: str
    color: str
    size: str


@pytest.mark.parametrize(
    ["value_object", "output"],
    [
        pytest.param(
            Apple("Granny Green", "green", "small"),
            {"is good": True},
        ),
        pytest.param(
            Apple("Granny Green", "red", "small"),
            {"is good": False},
        ),
        pytest.param(
            Apple("Juicy Red", "red", "small"),
            {"is good": True},
        ),
        pytest.param(
            Apple("Juicy Red", "red", "big"),
            {"is good": False},
        ),
        pytest.param(
            Apple("Big Red", "red", "big"),
            {"is good": True},
        ),
        pytest.param(
            Apple("Big Red", "green", "big"),
            {"is good": False},
        ),
        pytest.param(
            Apple("Big Red", "blue", "big"),
            {"is good": False},
        ),
        pytest.param(
            Apple("Big Red", "red", "small"),
            {"is good": False},
        ),
    ],
)
def test_tree_matcher_first_match_flat_single_values(
    value_object: Apple, output: dict[str, Any]
) -> None:
    tree: SolutionTree[Apple, dict[str, bool]] = SolutionTree(
        {
            "schema": {
                "selectors": {
                    "family": [
                        "Granny Green",
                        "Juicy Red",
                        "Big Red",
                    ],
                    "color": [
                        "green",
                        "red",
                        "blue",
                    ],
                    "size": [
                        "small",
                        "big",
                    ],
                },
                "output": {"is good": "bool"},
            },
            "apply first": [
                {
                    "when": {"family": ["Granny Green"], "color": ["green"]},
                    "set": {"is good": True},
                },
                {
                    "when": {
                        "family": "Juicy Red",
                        "color": "red",
                        "size": "small",
                    },
                    "set": {"is good": True},
                },
                {
                    "when": {
                        "family": "Big Red",
                        "color": "red",
                        "size": "big",
                    },
                    "set": {"is good": True},
                },
                {"when": {}, "set": {"is good": False}},
            ],
        },
        {
            "family": lambda apple: apple.family,
            "color": lambda apple: apple.color,
            "size": lambda apple: apple.size,
        },
    )

    assert tree.match_update(value_object) == output


@pytest.mark.parametrize(
    ["value_object", "output"],
    [
        pytest.param(
            Apple("Granny Green", "green", "small"),
            {
                "is good": True,
            },
        ),
        pytest.param(
            Apple("Granny Green", "red", "small"),
            {
                "is good": False,
            },
        ),
        pytest.param(
            Apple("Juicy Red", "red", "small"),
            {
                "is good": True,
            },
        ),
        pytest.param(
            Apple("Juicy Red", "red", "big"),
            {
                "is good": False,
            },
        ),
        pytest.param(
            Apple("Big Red", "red", "big"),
            {
                "is good": True,
            },
        ),
        pytest.param(
            Apple("Big Red", "green", "big"),
            {
                "is good": False,
            },
        ),
        pytest.param(
            Apple("Big Red", "blue", "big"),
            {
                "is good": False,
                "new type of apple": True,
            },
        ),
        pytest.param(
            Apple("Big Red", "red", "small"),
            {
                "is good": False,
            },
        ),
    ],
)
def test_tree_matcher_first_match_flat_single_values_2(
    value_object: Apple, output: dict[str, Any]
) -> None:
    tree: SolutionTree[Apple, dict[str, bool]] = SolutionTree(
        {
            "schema": {
                "selectors": {
                    "family": [
                        "Granny Green",
                        "Juicy Red",
                        "Big Red",
                    ],
                    "color": [
                        "green",
                        "red",
                        "blue",
                    ],
                    "size": [
                        "small",
                        "big",
                    ],
                },
                "output": {
                    "is good": "bool",
                    "new type of apple": "bool",
                },
            },
            "apply first": [
                {
                    "when": {"color": "blue"},
                    "set": {
                        "is good": False,
                        "new type of apple": True,
                    },
                },
                {
                    "when": {
                        "family": "Granny Green",
                        "color": "green",
                    },
                    "set": {
                        "is good": True,
                    },
                },
                {
                    "when": {
                        "family": "Juicy Red",
                        "color": "red",
                        "size": "small",
                    },
                    "set": {
                        "is good": True,
                    },
                },
                {
                    "when": {
                        "family": "Big Red",
                        "color": "red",
                        "size": "big",
                    },
                    "set": {
                        "is good": True,
                    },
                },
                {
                    "when": {},
                    "set": {
                        "is good": False,
                    },
                },
            ],
        },
        {
            "family": lambda apple: apple.family,
            "color": lambda apple: apple.color,
            "size": lambda apple: apple.size,
        },
    )

    assert tree.match_update(value_object) == output


@pytest.mark.parametrize(
    ["value_object", "output"],
    [
        pytest.param(
            Apple("Granny Green", "green", "small"),
            {
                "is good": True,
            },
        ),
        pytest.param(
            Apple("Granny Green", "green", "big"),
            {
                "is good": True,
            },
        ),
        pytest.param(
            Apple("Granny Green", "green", "extra"),
            {
                "is good": True,
            },
        ),
        pytest.param(
            Apple("Granny Green", "green", "ex-extra"),
            {
                "is good": True,
                "new type of apple": True,
            },
        ),
        pytest.param(
            Apple("Granny Green", "red", "small"),
            {
                "is good": False,
            },
        ),
        pytest.param(
            Apple("Granny Green", "red", "big"),
            {
                "is good": False,
            },
        ),
        pytest.param(
            Apple("Granny Green", "red", "extra"),
            {
                "is good": False,
            },
        ),
        pytest.param(
            Apple("Juicy Red", "red", "small"),
            {
                "is good": True,
            },
        ),
        pytest.param(
            Apple("Juicy Red", "red", "big"),
            {
                "is good": False,
            },
        ),
        pytest.param(
            Apple("Juicy Red", "red", "extra"),
            {
                "is good": False,
            },
        ),
        pytest.param(
            Apple("Big Red", "red", "small"),
            {
                "is good": False,
            },
        ),
        pytest.param(
            Apple("Big Red", "red", "big"),
            {
                "is good": True,
            },
        ),
        pytest.param(
            Apple("Big Red", "red", "extra"),
            {
                "is good": True,
            },
        ),
        pytest.param(
            Apple("Juicy Red", "blue", "big"),
            {
                "is good": False,
                "new type of apple": True,
            },
        ),
        pytest.param(
            Apple("Juicy Red", "blue", "extra"),
            {
                "is good": False,
                "new type of apple": True,
            },
        ),
        pytest.param(
            Apple("Big Red", "violet", "small"),
            {
                "is good": False,
                "new type of apple": True,
            },
        ),
        pytest.param(
            Apple("Big Red", "blue", "big"),
            {
                "is good": False,
                "new type of apple": True,
            },
        ),
        pytest.param(
            Apple("Big Red", "blue", "extra"),
            {
                "is good": False,
                "new type of apple": True,
            },
        ),
        pytest.param(
            Apple("Big Red", "violet", "extra"),
            {
                "is good": False,
                "new type of apple": True,
            },
        ),
        pytest.param(
            Apple("Strange Family", "blue", "big"),
            {
                "new type of apple": True,
                "unprocessable": True,
            },
        ),
        pytest.param(
            Apple("Strange Family", "blue", "extra"),
            {
                "new type of apple": True,
                "unprocessable": True,
            },
        ),
        pytest.param(
            Apple("Strange Family", "violet", "small"),
            {
                "new type of apple": True,
                "unprocessable": True,
            },
        ),
        pytest.param(
            Apple("Strange Family", "blue", "big"),
            {
                "new type of apple": True,
                "unprocessable": True,
            },
        ),
        pytest.param(
            Apple("Strange Family", "blue", "extra"),
            {
                "new type of apple": True,
                "unprocessable": True,
            },
        ),
        pytest.param(
            Apple("Strange Family", "violet", "extra"),
            {
                "new type of apple": True,
                "unprocessable": True,
            },
        ),
    ],
)
def test_tree_matcher_first_match_multi_values(
    value_object: Apple, output: dict[str, Any]
) -> None:
    tree: SolutionTree[Apple, dict[str, bool]] = SolutionTree(
        {
            "schema": {
                "selectors": {
                    "family": [
                        "Granny Green",
                        "Juicy Red",
                        "Big Red",
                        "Strange Family",
                    ],
                    "color": [
                        "green",
                        "red",
                        "blue",
                        "violet",
                    ],
                    "size": [
                        "small",
                        "big",
                        "extra",
                        "ex-extra",
                    ],
                },
                "output": {
                    "is good": "bool",
                    "new type of apple": "bool",
                },
            },
            "apply all": [
                {
                    "when": {"color": ["blue", "violet"]},
                    "set": {
                        "new type of apple": True,
                    },
                },
                {
                    "when": {
                        "family": ["Granny Green", "Juicy Red", "Big Red"],
                    },
                    "set": {
                        "is good": False,
                    },
                    "also": [
                        {
                            "when": {
                                "family": "Granny Green",
                                "color": "green",
                            },
                            "set": {
                                "is good": True,
                            },
                            "also": [
                                {
                                    "when": {"size": "ex-extra"},
                                    "set": {"new type of apple": True},
                                }
                            ],
                        },
                        {
                            "when": {
                                "family": "Juicy Red",
                                "color": "red",
                                "size": "small",
                            },
                            "set": {
                                "is good": True,
                            },
                        },
                        {
                            "when": {
                                "family": "Big Red",
                                "color": "red",
                                "size": ["big", "extra"],
                            },
                            "set": {
                                "is good": True,
                            },
                        },
                    ],
                },
                {
                    "when": {
                        "family": "Strange Family",
                    },
                    "set": {
                        "unprocessable": True,
                    },
                },
            ],
        },
        {
            "family": lambda apple: apple.family,
            "color": lambda apple: apple.color,
            "size": lambda apple: apple.size,
        },
    )

    assert tree.match_update(value_object) == output


@pytest.mark.parametrize(
    ["value_object", "output"],
    [
        pytest.param(
            Apple("Granny Green", "green", "small"),
            {
                "is good": True,
            },
        ),
        pytest.param(
            Apple("Granny Green", "green", "big"),
            {
                "is good": True,
            },
        ),
        pytest.param(
            Apple("Granny Green", "green", "extra"),
            {
                "is good": True,
            },
        ),
        pytest.param(
            Apple("Granny Green", "green", "ex-extra"),
            {
                "is good": True,
                "new type of apple": True,
            },
        ),
        pytest.param(
            Apple("Granny Green", "red", "small"),
            {
                "is good": False,
            },
        ),
        pytest.param(
            Apple("Granny Green", "red", "big"),
            {
                "is good": False,
            },
        ),
        pytest.param(
            Apple("Granny Green", "red", "extra"),
            {
                "is good": False,
            },
        ),
        pytest.param(
            Apple("Juicy Red", "red", "small"),
            {
                "is good": True,
            },
        ),
        pytest.param(
            Apple("Juicy Red", "red", "big"),
            {
                "is good": False,
            },
        ),
        pytest.param(
            Apple("Juicy Red", "red", "extra"),
            {
                "is good": False,
            },
        ),
        pytest.param(
            Apple("Big Red", "red", "small"),
            {
                "is good": False,
            },
        ),
        pytest.param(
            Apple("Big Red", "red", "big"),
            {
                "is good": True,
            },
        ),
        pytest.param(
            Apple("Big Red", "red", "extra"),
            {
                "is good": True,
            },
        ),
        pytest.param(
            Apple("Big Red", "blue", "extra"),
            {"is good": False, "new type of apple": True},
        ),
        pytest.param(
            Apple("Big Red", "violet", "extra"),
            {"is good": False, "new type of apple": True},
        ),
    ],
)
def test_tree_matcher_first_match_multi_values(
    value_object: Apple, output: dict[str, Any]
) -> None:
    tree: SolutionTree[Apple, dict[str, bool]] = SolutionTree(
        {
            "schema": {
                "selectors": {
                    "family": [
                        "Granny Green",
                        "Juicy Red",
                        "Big Red",
                        "Strange Family",
                    ],
                    "color": [
                        "green",
                        "red",
                        "blue",
                        "violet",
                    ],
                    "size": [
                        "small",
                        "big",
                        "extra",
                        "ex-extra",
                    ],
                },
                "output": {
                    "is good": "bool",
                    "new type of apple": "bool",
                    "unprocessable": "bool",
                },
            },
            "apply first": [
                {
                    "when": {
                        "family": ["Granny Green", "Juicy Red", "Big Red"],
                    },
                    "set": {
                        "is good": False,
                    },
                    "also": [
                        {
                            "when": {
                                "family": "Granny Green",
                                "color": "green",
                            },
                            "set": {
                                "is good": True,
                            },
                            "also": [
                                {
                                    "when": {"size": "ex-extra"},
                                    "set": {"new type of apple": True},
                                }
                            ],
                        },
                        {
                            "when": {
                                "family": "Juicy Red",
                                "color": "red",
                                "size": "small",
                            },
                            "set": {
                                "is good": True,
                            },
                        },
                        {
                            "when": {
                                "family": "Big Red",
                                "color": "red",
                                "size": ["big", "extra"],
                            },
                            "set": {
                                "is good": True,
                            },
                        },
                        {
                            "when": {"color": ["blue", "violet"]},
                            "set": {"new type of apple": True},
                        },
                    ],
                },
                {"when": {}, "set": {"unprocessable": True, "new type of apple": True}},
            ],
        },
        {
            "family": lambda apple: apple.family,
            "color": lambda apple: apple.color,
            "size": lambda apple: apple.size,
        },
    )

    assert tree.match_update(value_object) == output
