from dataclasses import dataclass
import re
from typing import Any, NamedTuple, Optional
import pytest

from src.parser import (
    Const,
    Enumerated,
    Identity,
    Opt,
    Parser,
    SyntaxPrasingError,
    Type,
    DictOf,
    ListOf,
    DictExp,
    UnionExp,
    Scope,
)


class Box:
    def __init__(self, data: Any) -> None:
        self.data = data

    def __eq__(self, value: object) -> bool:
        return type(value) is Box and self.data == value.data


class BoxA:
    def __init__(self, data: Any) -> None:
        self.data = data

    def __eq__(self, value: object) -> bool:
        return type(value) is BoxA and self.data == value.data


class BoxB:
    def __init__(self, data: Any) -> None:
        self.data = data

    def __eq__(self, value: object) -> bool:
        return type(value) is BoxB and self.data == value.data


@pytest.mark.parametrize(
    ["value", "parser", "expected"],
    [
        pytest.param(
            [1, 2, 3],
            Type(list),
            [1, 2, 3],
        ),
        pytest.param(
            1,
            Type(int),
            1,
        ),
        pytest.param(
            1,
            Type(int, Box),
            Box(1),
        ),
    ],
)
def test_type_expression_parser(value: Any, parser: Parser, expected: Any) -> None:
    assert parser.parse_value(value) == expected


@pytest.mark.parametrize(
    ["value", "parser", "expected"],
    [
        pytest.param(
            [1, 2, 3],
            Identity(Type(list), Box),
            Box([1, 2, 3]),
        ),
        pytest.param(
            1,
            Identity(Identity(Type(int), Box), Box),
            Box(Box(1)),
        ),
    ],
)
def test_identity_type_expression_parser(
    value: Any, parser: Parser, expected: Any
) -> None:
    assert parser.parse_value(value) == expected


@pytest.mark.parametrize(
    ["value", "parser"],
    [
        pytest.param(
            [1, 2, 3],
            Type(dict),
        ),
        pytest.param(
            1,
            Type(str),
        ),
        pytest.param(
            1,
            Type(float),
        ),
        pytest.param(
            "1",
            Type(int),
        ),
        pytest.param(
            0,
            Type(bool),
        ),
        pytest.param(
            1,
            Type(str, Box),
        ),
    ],
)
def test_type_expression_parser_fails(value: Any, parser: Parser) -> None:
    with pytest.raises(
        SyntaxPrasingError,
    ):
        parser.parse_value(value)


@pytest.mark.parametrize(
    ["value", "parser", "expected"],
    [
        pytest.param(
            [1, 2, 3],
            UnionExp(Const(None), Type(list)),
            [1, 2, 3],
        ),
        pytest.param(
            1,
            UnionExp(Const(None), Type(int)),
            1,
        ),
        pytest.param(
            1,
            UnionExp(Const(None), Type(int, Box)),
            Box(1),
        ),
        pytest.param(
            None,
            UnionExp(Const(None), Type(list)),
            None,
        ),
        pytest.param(
            None,
            UnionExp(Const(None), Type(int)),
            None,
        ),
        pytest.param(
            None,
            UnionExp(Const(None), Type(int, Box)),
            None,
        ),
    ],
)
def test_optional_type_expression_parser(
    value: Any, parser: Parser, expected: Any
) -> None:
    assert parser.parse_value(value) == expected


@pytest.mark.parametrize(
    ["value", "parser"],
    [
        pytest.param(
            [1, 2, 3],
            UnionExp(Const(None), Type(dict)),
        ),
        pytest.param(
            1,
            UnionExp(Const(None), Type(str)),
        ),
        pytest.param(
            1,
            UnionExp(Const(None), Type(float)),
        ),
        pytest.param(
            "1",
            UnionExp(Const(None), Type(int)),
        ),
        pytest.param(
            0,
            UnionExp(Const(None), Type(bool)),
        ),
        pytest.param(
            1,
            UnionExp(Const(None), Type(str, Box)),
        ),
    ],
)
def test_optional_type_expression_parser_fails(value: Any, parser: Parser) -> None:
    with pytest.raises(SyntaxPrasingError) as err:
        parser.parse_value(value)


@pytest.mark.parametrize(
    ["value", "parser", "expected"],
    [
        pytest.param(
            [1, 2, 3],
            ListOf(Type(int)),
            [1, 2, 3],
        ),
        pytest.param(
            [1],
            ListOf(Type(int), Box),
            Box([1]),
        ),
        pytest.param(
            [1],
            ListOf(Type(int, Box)),
            [Box(1)],
        ),
        pytest.param(
            [1],
            ListOf(Type(int, Box), Box),
            Box([Box(1)]),
        ),
        pytest.param(
            [None],
            ListOf(UnionExp(Const(None), Type(int))),
            [None],
        ),
        pytest.param(
            [None, 2, 3, None],
            ListOf(UnionExp(Const(None), Type(int))),
            [None, 2, 3, None],
        ),
        pytest.param(
            [None, 2, 3, None],
            ListOf(UnionExp(Const(None), Type(int, Box))),
            [None, Box(2), Box(3), None],
        ),
        pytest.param(
            None,
            UnionExp(Const(None), ListOf(Type(int))),
            None,
        ),
    ],
)
def test_list_of_type_expression_parser(
    value: Any, parser: Parser, expected: Any
) -> None:
    assert parser.parse_value(value) == expected


@pytest.mark.parametrize(
    ["value", "parser"],
    [
        pytest.param(
            [1, 2, 3],
            ListOf(Type(str)),
        ),
        pytest.param(
            [1],
            ListOf(Type(float), Box),
        ),
        pytest.param(
            ["1", "2"],
            ListOf(Type(int, Box)),
        ),
        pytest.param(
            [1, "2", 3],
            ListOf(Type(int, Box), Box),
        ),
        pytest.param(
            [None],
            ListOf(Type(int)),
        ),
        pytest.param(
            None,
            ListOf(Type(int)),
        ),
    ],
)
def test_list_of_type_expression_parser_fails(value: Any, parser: Parser) -> None:
    with pytest.raises(SyntaxPrasingError) as err:
        parser.parse_value(value)


@pytest.mark.parametrize(
    ["value", "parser"],
    [
        pytest.param(
            [1, 2, 3],
            ListOf(Type(str)),
        ),
        pytest.param(
            [1],
            ListOf(Type(float), Box),
        ),
        pytest.param(
            ["1", "2"],
            ListOf(Type(int, Box)),
        ),
        pytest.param(
            [1, "2", 3],
            ListOf(Type(int, Box), Box),
        ),
        pytest.param(
            [None],
            ListOf(Type(int)),
        ),
        pytest.param(
            None,
            ListOf(Type(int)),
        ),
    ],
)
def test_enum_type_expression_parser_fails(value: Any, parser: Parser) -> None:
    with pytest.raises(SyntaxPrasingError) as err:
        parser.parse_value(value)


@pytest.mark.parametrize(
    ["value", "parser", "expected"],
    [
        pytest.param(
            {"a": 1, "b": 2, "c": 3},
            DictOf(Type(int)),
            {"a": 1, "b": 2, "c": 3},
        ),
        pytest.param(
            {"a": 1},
            DictOf(Type(int), Box),
            Box({"a": 1}),
        ),
        pytest.param(
            {"a": 1},
            DictOf(Type(int, Box)),
            {"a": Box(1)},
        ),
        pytest.param(
            {"a": 1},
            DictOf(Type(int, Box), Box),
            Box({"a": Box(1)}),
        ),
        pytest.param(
            {"a": None},
            DictOf(UnionExp(Const(None), Type(int))),
            {"a": None},
        ),
        pytest.param(
            {},
            DictOf(Type(int)),
            {},
        ),
        pytest.param(
            {"a": None, "b": 2, "c": 3, "d": None},
            DictOf(UnionExp(Const(None), Type(int))),
            {"a": None, "b": 2, "c": 3, "d": None},
        ),
        pytest.param(
            {"a": None, "b": 2, "c": 3, "d": None},
            DictOf(UnionExp(Const(None), Type(int, Box))),
            {"a": None, "b": Box(2), "c": Box(3), "d": None},
        ),
        pytest.param(
            {"a": None, "b": 2, "c": 3, "d": None},
            DictOf(
                UnionExp(Const(None), Type(int)),
                key_is_allowed=lambda key: len(key) == 1,
            ),
            {"a": None, "b": 2, "c": 3, "d": None},
        ),
        pytest.param(
            {"a": {"b": "2", "c": "3"}},
            DictOf(DictOf(Type(str, Box))),
            {"a": {"b": Box("2"), "c": Box("3")}},
        ),
        pytest.param(
            None,
            UnionExp(Const(None), DictOf(Type(int))),
            None,
        ),
    ],
)
def test_dict_of_type_expression_parser(
    value: Any, parser: Parser, expected: Any
) -> None:
    assert parser.parse_value(value) == expected


@pytest.mark.parametrize(
    ["value", "parser"],
    [
        pytest.param(
            [1, 2, 3],
            DictOf(Type(str)),
        ),
        pytest.param(
            [1],
            DictOf(Type(float), Box),
        ),
        pytest.param(
            ["1", "2"],
            DictOf(Type(int, Box)),
        ),
        pytest.param(
            [1, "2", 3],
            DictOf(Type(int, Box), Box),
        ),
        pytest.param(
            [None],
            DictOf(Type(int)),
        ),
        pytest.param(
            None,
            DictOf(Type(int)),
        ),
        pytest.param(
            {"a": 1, "b": "2", "c": 3},
            DictOf(Type(int)),
        ),
        pytest.param(
            {"a": "1"},
            DictOf(Type(int), Box),
        ),
        pytest.param(
            {"a": "1"},
            DictOf(Type(int, Box)),
        ),
        pytest.param(
            {"a": 1},
            DictOf(Type(str, Box), Box),
        ),
        pytest.param(
            {"a": 1},
            DictOf(Type(float, Box), Box),
        ),
        pytest.param(
            {"a": None, "b": 2, "c": 3, "d": None},
            DictOf(UnionExp(Const(None), Type(str))),
        ),
        pytest.param(
            {"a": None, "b": [1, 2], "c": [3]},
            DictOf(UnionExp(Const(None), ListOf(Type(str)))),
        ),
        pytest.param(
            {"a": None, "b": 2, "c": 3, "d": None},
            DictOf(
                UnionExp(Const(None), Type(int)),
                key_is_allowed=lambda key: len(key) == 2,
            ),
        ),
        pytest.param(
            {"a": None, "b": 2, "c": 3, "d": None},
            DictOf(
                UnionExp(Const(None), Type(int)),
                key_is_allowed=lambda key: key in {"a", "b", "c"},
            ),
        ),
    ],
)
def test_dict_of_type_expression_parser_fails(value: Any, parser: Parser) -> None:
    with pytest.raises(SyntaxPrasingError) as err:
        parser.parse_value(value)


@pytest.mark.parametrize(
    ["value", "parser", "expected"],
    [
        pytest.param(
            [1, 2, 3],
            UnionExp(Type(list), Type(int)),
            [1, 2, 3],
        ),
        pytest.param(
            2,
            UnionExp(Type(list), Type(int)),
            2,
        ),
        pytest.param(
            [1, 2, 3],
            UnionExp(Type(list, Box), Type(int)),
            Box([1, 2, 3]),
        ),
        pytest.param(
            2,
            UnionExp(Type(list), Type(int, Box)),
            Box(2),
        ),
        pytest.param(
            [1, 2, 3],
            UnionExp(Type(list), Type(int, Box)),
            [1, 2, 3],
        ),
        pytest.param(
            2,
            UnionExp(Type(list, Box), Type(int)),
            2,
        ),
        pytest.param(
            [1, 2, 3],
            UnionExp(ListOf(Type(int, Box)), Type(int, Box)),
            [Box(1), Box(2), Box(3)],
        ),
        pytest.param(
            1,
            UnionExp(Type(int)),
            1,
        ),
        pytest.param(
            1,
            UnionExp(Type(int, Box)),
            Box(1),
        ),
        pytest.param(
            {"b": 2, "c": "3", "d": [1, 2, 3]},
            DictOf(UnionExp(Type(int), Type(str), ListOf(Type(int)))),
            {"b": 2, "c": "3", "d": [1, 2, 3]},
        ),
        pytest.param(
            [1, "2", 3],
            ListOf(UnionExp(Type(int, BoxA), Type(str, BoxB))),
            [BoxA(1), BoxB("2"), BoxA(3)],
        ),
    ],
)
def test_union_type_expression_parser(
    value: Any, parser: Parser, expected: Any
) -> None:
    assert parser.parse_value(value) == expected


@pytest.mark.parametrize(
    ["value", "parser"],
    [
        pytest.param(
            [1, "2", 3],
            UnionExp(ListOf(Type(int)), Type(int)),
        ),
        pytest.param(
            "2",
            UnionExp(ListOf(Type(int)), Type(int)),
        ),
        pytest.param(
            2.0,
            UnionExp(Type(list), Type(int, Box)),
        ),
        pytest.param(
            "1",
            UnionExp(Type(int)),
        ),
        pytest.param(
            {"b": 2, "c": "3", "d": [1, 2, "3"]},
            DictOf(UnionExp(Type(int), Type(str), ListOf(Type(int)))),
        ),
        pytest.param(
            {"b": None, "b": 2, "c": "3", "d": [1, 2, "3"]},
            DictOf(UnionExp(Type(int), Type(str), ListOf(Type(int)))),
        ),
        pytest.param(
            [1, "2", 3.0],
            ListOf(UnionExp(Type(int, BoxA), Type(str, BoxB))),
        ),
    ],
)
def test_union_type_expression_parser_fails(value: Any, parser: Parser) -> None:
    with pytest.raises(
        SyntaxPrasingError,
    ):
        parser.parse_value(value)


@pytest.mark.parametrize(
    ["value", "parser", "expected"],
    [
        pytest.param(
            {"a": [1, 2, 3], "b": 123},
            DictExp(
                {
                    "a": UnionExp(Type(list), Type(int)),
                    "b": UnionExp(Type(list), Type(int)),
                }
            ),
            {"a": [1, 2, 3], "b": 123},
        ),
        pytest.param(
            {"a": 2},
            DictExp(
                {
                    "a": UnionExp(Type(list), Type(int)),
                }
            ),
            {"a": 2},
        ),
        pytest.param(
            {"a": 2},
            DictExp(
                {
                    "a": UnionExp(Type(list, Box), Type(int, Box)),
                }
            ),
            {"a": Box(2)},
        ),
        pytest.param(
            {"a": 2},
            DictExp(
                {
                    "a": Opt(Type(int, Box)),
                }
            ),
            {"a": Box(2)},
        ),
        pytest.param(
            {},
            DictExp(
                {
                    "a": Opt(Type(int, Box)),
                }
            ),
            {},
        ),
        pytest.param(
            {"a": 2},
            DictExp(
                {
                    "a": UnionExp(
                        ListOf(Enumerated([1, 2, 3]), Box),
                        Enumerated([1, 2, 3], Box),
                    ),
                }
            ),
            {"a": Box(2)},
        ),
        pytest.param(
            {"a": None},
            DictExp({"a": Enumerated([1, 2, 3, None], Box)}),
            {"a": Box(None)},
        ),
    ],
)
def test_dict_type_expression_parser(value: Any, parser: Parser, expected: Any) -> None:
    assert parser.parse_value(value) == expected


@pytest.mark.parametrize(
    ["value", "parser"],
    [
        pytest.param(
            {"a": [1, 2, 3], "b": 123},
            DictExp(
                {
                    "a": UnionExp(Type(list), Type(int)),
                }
            ),
        ),
        pytest.param(
            {"a": 2},
            DictExp(
                {
                    "a": UnionExp(Type(list), Type(str)),
                }
            ),
        ),
        pytest.param(
            {"a": 2},
            DictExp(
                {
                    "a": UnionExp(Type(list, Box), Type(str, Box)),
                }
            ),
        ),
        pytest.param(
            {"a": 20},
            DictExp(
                {
                    "a": UnionExp(
                        ListOf(Enumerated([1, 2, 3]), Box),
                        Enumerated([1, 2, 3], Box),
                    ),
                }
            ),
        ),
    ],
)
def test_dict_type_expression_parser_fails(value: Any, parser: Parser) -> None:
    with pytest.raises(
        SyntaxPrasingError,
    ):
        parser.parse_value(value)


def test_scoped_type_expression_parser() -> None:
    scope = Scope(
        "Tree",
        parser_assembler=lambda scoped: {
            "Node": UnionExp(
                Type(int),
                DictExp(
                    {
                        "Left": scoped("Node"),
                        "Right": Opt(scoped("Node")),
                    }
                ),
                DictExp(
                    {
                        "Left": Opt(scoped("Node")),
                        "Right": scoped("Node"),
                    }
                ),
            )
        },
    )
    Node = scope.get_scoped_parser("Node")

    assert Node.parse_value({"Left": {"Right": 0}, "Right": {"Right": 1}}) == {
        "Left": {"Right": 0},
        "Right": {"Right": 1},
    }

    assert Node.parse_value(
        {
            "Left": {"Right": 0, "Left": {"Left": 0}},
            "Right": {"Right": 1},
        }
    ) == {
        "Left": {"Right": 0, "Left": {"Left": 0}},
        "Right": {"Right": 1},
    }

    with pytest.raises(SyntaxPrasingError):
        Node.parse_value({"Left": {"Right": "0"}, "Right": {"Right": 1}})

    with pytest.raises(SyntaxPrasingError):
        Node.parse_value({"Left": {"a": 0}, "Right": {"Right": 1}})

    with pytest.raises(SyntaxPrasingError):
        Node.parse_value({"Right": {"Left": [1]}})

    with pytest.raises(SyntaxPrasingError):
        Node.parse_value(
            {
                "Left": {"Right": 0, "Left": None},
                "Right": {"Right": 1, "Left": None},
            }
        )

    with pytest.raises(SyntaxPrasingError):
        Node.parse_value(
            {
                "Left": {"Right": 0, "Left": {"Right": None, "Left": 0}},
                "Right": {"Right": 1, "Left": None},
            }
        )
