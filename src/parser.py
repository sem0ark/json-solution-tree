from functools import partial
from typing import (
    Any,
    Callable,
    NamedTuple,
    Optional,
    Protocol,
    Union,
)

from abc import ABC, abstractmethod
from pprint import pformat


def show_part(target: Any, max_len=200) -> str:
    representation = str(target)
    if len(representation) < 200:
        return representation
    return representation[:200] + "..."


def identity(target: Any) -> Any:
    return target


class SyntaxPrasingError(ValueError): ...


class Parser(ABC):
    @abstractmethod
    def parse_value(self, target, parse_blindly=False) -> Any:
        """Check that value is matching and attempt to compile it to an object via constructor.
        Uses `parse_blindly` to avoid matching the structure multiple times.
        """

    @abstractmethod
    def is_matching(self, target, parse_blindly: bool) -> bool:
        """Checks whether some value is matching the specified structure."""

    @abstractmethod
    def get_syntax_string(self, continue_: bool) -> str:
        """Returns a string representation of the syntax structure for readability purposes.

        `continue_` is used to trim nesting and avoid recursion
        """

    def raise_parse_error(self, target):
        raise SyntaxPrasingError(
            f"Failed to parse {show_part(target)}, expected {self.get_syntax_string(True)}"
        )


class Type(Parser):
    def __init__(self, type_: Any, constructor: Callable[[Any], Any] = None) -> None:
        self.type = type_
        self.constructor = constructor or (identity)

    def is_matching(self, target: Any) -> bool:
        return type(target) is self.type

    def get_syntax_string(self, continue_=False) -> str:
        return f"{self.type.__name__}"

    def parse_value(self, target: Any, parse_blindly=False) -> Any:
        if not self.is_matching(target):
            self.raise_parse_error(target)

        return self.constructor(target)


class Enumerated(Parser):
    def __init__(self, *values: Any, constructor: Callable[[Any], Any] = None) -> None:
        self.values = set(values)
        self.types = {type(x).__qualname__ for x in values}
        self.constructor = constructor or (identity)

    def is_matching(self, target: Any) -> bool:
        if type(target).__qualname__ not in self.types:
            return False

        return target in self.values

    def get_syntax_string(self, continue_=False) -> str:
        return f"enum{pformat(self.values)}"

    def parse_value(self, target: Any, parse_blindly=False) -> Any:
        if not self.is_matching(target):
            self.raise_parse_error(target)

        return self.constructor(target)


class Const(Parser):
    def __init__(self, value: Any, constructor: Callable[[Any], Any] = None) -> None:
        self.value = value
        self.constructor = constructor or (identity)

    def is_matching(self, target: Any) -> bool:
        return target == self.value

    def get_syntax_string(self, continue_=False) -> str:
        return f"const{self.value}"

    def parse_value(self, target: Any, parse_blindly=False) -> Any:
        if not self.is_matching(target):
            self.raise_parse_error(target)

        return self.constructor(target)


class DictOf(Parser):
    def __init__(
        self,
        matcher: Parser,
        constructor: Callable[[dict], Any] = None,
        key_is_allowed: Optional[Callable[[str], bool]] = None,
    ) -> None:
        self.matcher = matcher
        self.constructor = constructor or (identity)
        self.key_is_allowed = key_is_allowed or (lambda _: True)

    def is_matching(self, target: Any) -> bool:
        if type(target) is not dict:
            return False

        if not all(self.key_is_allowed(key) for key in target):
            return False

        return all(self.matcher.is_matching(value) for value in target.values())

    def get_syntax_string(self, continue_=False) -> str:
        return "{ [str]: " + self.matcher.get_syntax_string(continue_) + " }"

    def parse_value(self, target: Any, parse_blindly=False) -> Any:
        if not parse_blindly and not self.is_matching(target):
            message = f"Failed to parse {show_part(target)}, expected {self.get_syntax_string(True)}"
            additional_messages = []

            if type(target) is dict:
                additional_messages = [
                    f'Unexpected key "{key}"'
                    for key in target.keys()
                    if not self.key_is_allowed(key)
                ]
                if additional_messages:
                    message += "(" + ", ".join(additional_messages) + ")"

                raise SyntaxPrasingError(message)

            self.raise_parse_error(target)

        return self.constructor(
            {
                key: self.matcher.parse_value(value, parse_blindly=True)
                for key, value in target.items()
            }
        )


class ListOf(Parser):
    def __init__(
        self, matcher: Parser, constructor: Callable[[list], Any] = None
    ) -> None:
        self.matcher = matcher
        self.constructor = constructor or (identity)

    def is_matching(self, target: Any) -> bool:
        if type(target) is not list:
            return False

        return all(self.matcher.is_matching(v) for v in target)

    def get_syntax_string(self, continue_=False) -> str:
        return f"[{self.matcher.get_syntax_string(continue_)}]"

    def parse_value(self, target: Any, parse_blindly=False) -> Any:
        if not parse_blindly and not self.is_matching(target):
            self.raise_parse_error(target)

        return self.constructor(
            [self.matcher.parse_value(value, parse_blindly=True) for value in target]
        )


class Opt(NamedTuple):
    matcher: Parser


class DictExp(Parser):
    def __init__(
        self,
        types_dict: dict[Any, Union[Parser, Opt]],
        constructor: Callable[[dict], Any] = None,
    ) -> None:
        self.types_dict = types_dict
        self.constructor = constructor or (identity)

    def is_matching(self, target) -> bool:
        if type(target) is not dict:
            return False

        for key in target.keys():
            if key not in self.types_dict:
                return False

        for key, matcher in self.types_dict.items():
            if type(matcher) is not Opt and key not in target:
                return False

        for key, matcher in self.types_dict.items():
            if type(matcher) is Opt:
                if key not in target:
                    continue

                if not matcher.matcher.is_matching(target[key]):
                    return False

            elif not matcher.is_matching(target[key]):
                return False

        return True

    def get_syntax_string(self, continue_=False) -> str:
        return (
            "{ "
            + ", ".join(
                [
                    f"{key}: ?({matcher.matcher.get_syntax_string(continue_)})"
                    if type(matcher) is Opt
                    else f"{key}: {matcher.get_syntax_string(continue_)}"
                    for key, matcher in self.types_dict.items()
                ]
            )
            + " }"
        )

    def parse_value(self, target: Any, parse_blindly=False) -> Any:
        if not parse_blindly and not self.is_matching(target):
            self.raise_parse_error(target)

        return self.constructor(
            {
                key: (
                    matcher.matcher.parse_value(target[key], parse_blindly=True)
                    if type(matcher) is Opt
                    else matcher.parse_value(target[key], parse_blindly=True)
                )
                for key, matcher in self.types_dict.items()
                if key in target
            }
        )


class Identity(Parser):
    def __init__(
        self, matcher: Parser, constructor: Callable[[Any], Any] = None
    ) -> None:
        self.matcher = matcher
        self.constructor = constructor or (identity)

    def is_matching(self, target: Any) -> bool:
        return self.matcher.is_matching(target)

    def get_syntax_string(self, continue_=False) -> str:
        return self.matcher.get_syntax_string(continue_)

    def parse_value(self, target: Any, parse_blindly=False) -> Any:
        if not self.is_matching(target):
            self.raise_parse_error(target)

        return self.constructor(self.matcher.parse_value(target, parse_blindly=True))


class UnionExp(Parser):
    def __init__(
        self, *matchers: Parser, constructor: Callable[[Any], Any] = None
    ) -> None:
        self.matchers = matchers
        self.constructor = constructor or (identity)

    def is_matching(self, target: Any) -> bool:
        return any(matcher.is_matching(target) for matcher in self.matchers)

    def get_syntax_string(self, continue_=False) -> str:
        return " | ".join([m.get_syntax_string(continue_) for m in self.matchers])

    def parse_value(self, target: Any, parse_blindly=False) -> Any:
        for matcher in self.matchers:
            if matcher.is_matching(target):
                return self.constructor(matcher.parse_value(target, parse_blindly=True))
        else:
            self.raise_parse_error(target)


class Scoped(Parser):
    def __init__(
        self, scope: "Scope", name: str, constructor: Callable[[Any], Any] = None
    ) -> None:
        self.scope = scope
        self.name = name
        self.constructor = constructor or (identity)

    def is_matching(self, target: Any) -> bool:
        return self.scope.get_scoped_parser(self.name).is_matching(target)

    def get_syntax_string(self, continue_=False) -> str:
        if not continue_:
            return f"{self.scope.scope_name}::{self.name}"

        return (
            f"{self.scope.scope_name}::{self.name} = "
            + self.scope.get_scoped_parser(self.name).get_syntax_string()
        )

    def parse_value(self, target: Any, parse_blindly=False) -> Any:
        if not self.is_matching(target):
            self.raise_parse_error(target)

        return self.constructor(
            self.scope.get_scoped_parser(self.name).parse_value(target, parse_blindly)
        )


class ScopeAssembler(Protocol):
    def __call__(
        self, name: str, constructor: Callable[[Any], Any] = None, /
    ) -> dict: ...


class Scope:
    def __init__(self, scope_name: str, *, parser_assembler: ScopeAssembler) -> None:
        self.scope_name = scope_name
        self.types_dict = parser_assembler(partial(Scoped, self))

    def get_scoped_parser(self, name: str) -> Scoped:
        if name not in self.types_dict:
            raise ValueError(
                f"Scoped Parser construction failed, {self.scope_name}::{name} does not exist."
            )

        return self.types_dict[name]
