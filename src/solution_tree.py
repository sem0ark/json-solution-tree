from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Optional,
    Protocol,
    TypeVar,
    Union,
    cast,
)

import json

from pathlib import Path
from functools import wraps

from src.parser import (
    Parser,
    Identity,
    Opt,
    Scope,
    Type,
    Enumerated,
    Const,
    Opt,
    DictOf,
    ListOf,
    DictExp,
    UnionExp,
    Scope,
)


def cache_function(
    func: Callable[[Any], Any], cache_dict: dict[str, Any], cache_key: str
) -> Callable[[Any], Any]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: dict[str, Any]) -> Any:
        if cache_key in cache_dict:
            return cache_dict[cache_key]
        return cache_dict.setdefault(cache_key, func(*args, **kwargs))

    return wrapper


def check_json_path(json_config_path: Path) -> bool:
    if not json_config_path.exists():
        raise ValueError(f"Configuration does not exist on path {json_config_path}")

    if not json_config_path.is_file():
        raise ValueError(f"{json_config_path} expected to be a file")

    if not json_config_path.match("*.json"):
        raise ValueError(f"{json_config_path} expected to be a JSON file")

    return True


Object_Type = TypeVar("Object_Type")
Output_Type = TypeVar("Output_Type", bound=dict[str, Any])


class Matcher(Protocol, Generic[Object_Type]):
    def intersect(
        self, other: "Matcher[Object_Type]"
    ) -> Optional["Matcher[Object_Type]"]: ...

    def match(self, value: Object_Type) -> bool: ...

    def __repr__(self) -> str:
        return "Matcher"


T = TypeVar("T")


class ValueMatcher(Generic[T, Object_Type], Matcher[Object_Type]):
    def __init__(
        self, selector: Callable[[Object_Type], T], values: Iterable[T]
    ) -> None:
        self._selector = selector
        self._values = set(values)
        self._is_negative = False

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is ValueMatcher
            and self._is_negative == other._is_negative
            and self._values == other._values
        )

    def __repr__(self) -> str:
        return f"ValueMatcher({self._values} {str(self._selector)[80:]})"

    @property
    def is_empty(self) -> bool:
        return len(self._values) == 0

    def match(self, value: Object_Type) -> bool:
        return self._selector(value) in self._values

    def intersect(self, other: Matcher[Object_Type]) -> Optional[Matcher[Object_Type]]:
        if type(other) is not ValueMatcher:
            return None

        new_matcher = ValueMatcher(self._selector, list(self._values & other._values))

        if new_matcher.is_empty:
            return None

        return new_matcher


class Query(Generic[Object_Type]):
    def __init__(
        self,
        matchers: dict[str, Matcher[Object_Type]],
        selectors: dict[str, Callable[[Object_Type], Any]],
    ) -> None:
        self.matchers = matchers
        self.selectors = selectors

    def __eq__(self, other: object) -> bool:
        if type(other) is not Query:
            return False

        if len(self.matchers) != len(other.matchers):
            return False

        for key, matcher in self.matchers.items():
            if key not in other.matchers or matcher != other.matchers[key]:
                return False

        return True

    def __repr__(self) -> str:
        return f"Query {self.matchers}"

    def intersect(self, other: "Query[Object_Type]") -> Optional["Query[Object_Type]"]:
        matchers = {**self.matchers}
        for key, matcher in other.matchers.items():
            if key in matchers:
                intersection = matchers[key].intersect(matcher)
                if intersection is None:
                    return None

                matchers[key] = intersection
            else:
                matchers[key] = matcher
        return Query(matchers, self.selectors)

    def match(self, value: Object_Type) -> bool:
        for matcher in self.matchers.values():
            if not matcher.match(value):
                return False

        return True


class Setter(Generic[Output_Type]):
    def __init__(self, update_dict: dict[str, Any]) -> None:
        self.update_dict = update_dict

    def __eq__(self, other: object) -> bool:
        return type(other) is Setter and self.update_dict == other.update_dict

    def __repr__(self) -> str:
        return f"Setter[{self.update_dict}]"

    def update(self, outer_dict: Output_Type) -> None:
        outer_dict.update(self.update_dict)


class Condition(Generic[Object_Type, Output_Type]):
    def __init__(
        self,
        query: Query[Object_Type],
        setter: Optional[Setter[Output_Type]],
        annotation: Optional[str],
        subconditions: Optional[
            Union[
                "SwitchApplyFirst[Object_Type, Output_Type]",
                "SwitchApplyAll[Object_Type, Output_Type]",
            ]
        ],
    ) -> None:
        self.query = query
        self.setter = setter
        self.subconditions = subconditions
        self.annotation = annotation

    def __repr__(self) -> str:
        return f"Condition {self.annotation or ''} [{self.query}, {self.setter}]"

    def match(self, value: Object_Type, output: Output_Type) -> bool:
        if self.query.match(value):
            if self.setter is not None:
                self.setter.update(output)

            if self.subconditions is not None:
                self.subconditions.match(value, output)

            return True

        return False


class SwitchApplyFirst(Generic[Object_Type, Output_Type]):
    def __init__(self, conditions: list["Condition[Object_Type, Output_Type]"]) -> None:
        self.conditions = conditions

    def match(self, value: Object_Type, output: Output_Type) -> bool:
        for condition in self.conditions:
            if condition.match(value, output):
                return True

        return False


class SwitchApplyAll(Generic[Object_Type, Output_Type]):
    def __init__(self, conditions: list["Condition[Object_Type, Output_Type]"]) -> None:
        self.conditions = conditions

    def match(self, value: Object_Type, output: Output_Type) -> bool:
        result = False
        for condition in self.conditions:
            if condition.match(value, output):
                result = True

        return result


class SolutionTree(Generic[Object_Type, Output_Type]):
    def __init__(
        self,
        json_config: dict[str, Any],
        selectors: dict[str, Callable[[Object_Type], Any]],
        test_schema=False,
    ) -> None:
        self.selectors_cache: dict[str, Any] = {}
        self.selectors = {
            key: cache_function(selector, self.selectors_cache, key)
            for key, selector in selectors.items()
        }
        self.tree = self.parse(json_config, self.selectors, test_schema)

    def match_update(self, value: Object_Type) -> Output_Type:
        result = cast(Output_Type, {})
        self.tree.match(value, result)
        return result

    @staticmethod
    def from_file(
        json_config_path: Path, selectors: dict[str, Callable[[Object_Type], Any]]
    ) -> "SolutionTree[Object_Type, Output_Type]":
        check_json_path(json_config_path)
        return SolutionTree(json.load(json_config_path.open()), selectors)

    def parse(
        self,
        json_config: dict[str, Any],
        selectors: dict[str, Callable[[Object_Type], Any]],
        parse_blindly=False,
    ):
        WHEN_CLAUSE = "when"
        ALSO_CLAUSE = "also"
        SET_CLAUSE = "set"
        ANNOTATION_CLAUSE = "_annotation"

        SWICTH_FIRST_MATCH = "apply first"
        SWICTH_ALL_MATCH = "apply all"

        schema, tree = (
            Scope(
                "Full",
                parser_assembler=lambda scoped: {
                    "Schema": DictExp(
                        {
                            "selectors": Type(dict),
                            "output": Type(dict),
                        }
                    ),
                    "root": UnionExp(
                        DictExp(
                            {
                                "schema": scoped("Schema"),
                                SWICTH_FIRST_MATCH: Type(list),
                            },
                            lambda d: (
                                d["schema"],
                                {SWICTH_FIRST_MATCH: d[SWICTH_FIRST_MATCH]},
                            ),
                        ),
                        DictExp(
                            {
                                "schema": scoped("Schema"),
                                SWICTH_ALL_MATCH: Type(list),
                            },
                            lambda d: (
                                d["schema"],
                                {SWICTH_ALL_MATCH: d[SWICTH_ALL_MATCH]},
                            ),
                        ),
                    ),
                },
            )
            .get_scoped_parser("root")
            .parse_value(json_config, parse_blindly)
        )

        selectors_schema, output_schema = (
            Scope(
                "Schema",
                parser_assembler=lambda scoped: {
                    "bool_type": Const("bool", constructor=lambda _: Type(bool)),
                    "str_type": Const("str", constructor=lambda _: Type(str)),
                    "number_type": Const(
                        "number", constructor=lambda _: UnionExp(Type(int), Type(float))
                    ),
                    "enum": ListOf(
                        UnionExp(
                            Const(None), Type(str), Type(int), Type(float), Type(bool)
                        ),
                        constructor=lambda values: Enumerated(values=values),
                    ),
                    "array": DictExp(
                        {
                            "list of": UnionExp(
                                scoped("bool_type"),
                                scoped("str_type"),
                                scoped("number_type"),
                                scoped("enum"),
                            )
                        },
                        constructor=lambda d: ListOf(d["list of"]),
                    ),
                    "root": DictExp(
                        {
                            "selectors": DictOf(
                                UnionExp(
                                    scoped("bool_type"),
                                    scoped("str_type"),
                                    scoped("number_type"),
                                    scoped("enum"),
                                )
                            ),
                            "output": DictOf(
                                UnionExp(
                                    scoped("bool_type"),
                                    scoped("str_type"),
                                    scoped("number_type"),
                                    scoped("enum"),
                                    scoped("array"),
                                )
                            ),
                        },
                        lambda d: (d["selectors"], d["output"]),
                    ),
                },
            )
            .get_scoped_parser("root")
            .parse_value(schema, parse_blindly)
        )

        for selector_name in selectors_schema:
            if selector_name not in selectors:
                raise ValueError(f'"{selector_name}" is required by the schema.')

        solution_tree_scope = Scope(
            "SolutionTree",
            parser_assembler=lambda scoped: {
                "SwitchFirst": DictExp(
                    {SWICTH_FIRST_MATCH: ListOf(scoped("Condition"))},
                    lambda d: SwitchApplyFirst(d[SWICTH_FIRST_MATCH]),
                ),
                "SwitchAll": DictExp(
                    {SWICTH_ALL_MATCH: ListOf(scoped("Condition"))},
                    lambda d: SwitchApplyAll(d[SWICTH_ALL_MATCH]),
                ),
                "WhenClause": DictExp(
                    {
                        name: Opt(
                            UnionExp(
                                Identity(
                                    _type,  # Turns out Python does not create scope for anonymous functions
                                    cast(
                                        Callable[[Any], ValueMatcher],
                                        lambda x, s=selectors[name]: ValueMatcher(
                                            s, [x]
                                        ),
                                    ),
                                ),
                                ListOf(
                                    _type,
                                    cast(
                                        Callable[[list[Any]], ValueMatcher],
                                        lambda x, s=selectors[name]: ValueMatcher(s, x),
                                    ),
                                ),
                            )
                        )
                        for name, _type in selectors_schema.items()
                    },
                    lambda matchers: Query(matchers, selectors),
                ),
                "SetClause": DictExp(
                    {
                        output_name: Opt(output_type)
                        for output_name, output_type in output_schema.items()
                    },
                    lambda update_dict: Setter(update_dict),
                ),
                "Condition": DictExp(
                    {
                        ANNOTATION_CLAUSE: Opt(Type(str)),
                        WHEN_CLAUSE: scoped("WhenClause"),
                        SET_CLAUSE: scoped("SetClause"),
                        ALSO_CLAUSE: Opt(
                            UnionExp(
                                ListOf(
                                    scoped("Condition"),
                                    lambda conditions: SwitchApplyFirst(
                                        cast(list[Condition], conditions)
                                    ),
                                ),
                                scoped("SwitchAll"),
                                scoped("SwitchFirst"),
                            )
                        ),
                    },
                    lambda d: Condition(
                        query=d[WHEN_CLAUSE],
                        setter=d.get(SET_CLAUSE, None),
                        annotation=d.get(ANNOTATION_CLAUSE, None),
                        subconditions=d.get(ALSO_CLAUSE, None),
                    ),
                ),
            },
        )

        return UnionExp(
            solution_tree_scope.get_scoped_parser("SwitchAll"),
            solution_tree_scope.get_scoped_parser("SwitchFirst"),
        ).parse_value(tree)
