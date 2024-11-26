from functools import partial
from typing import (
    Any,
    Callable,
    Optional,
    Union,
)

from textwrap import indent
from abc import ABC, abstractmethod
from pprint import pformat


def show_part(target: Any, max_len=100) -> str:
    representation = str(target)
    if len(representation) < max_len:
        return representation
    return representation[:max_len] + "..."


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
    def is_matching(self, target, shallow = False) -> bool:
        """Checks whether some value is matching the specified structure."""

    @abstractmethod
    def get_syntax_string(self, continue_: bool) -> str:
        """Returns a string representation of the syntax structure for readability purposes.

        `continue_` is used to trim nesting and avoid recursion
        """

    def raise_parse_error(self, target, additional_messages: list[str] | None = None):
        raise SyntaxPrasingError(
            f"Failed to parse {show_part(target)}, expected \n{self.get_syntax_string(True)}"
            + ("" if additional_messages is None
            else ("\n" + "\n".join(additional_messages)))
        )


class Type(Parser):
    def __init__(
        self, type_: Any, constructor: Optional[Callable[[Any], Any]] = None
    ) -> None:
        self.type = type_
        self.constructor = constructor or (identity)

    def is_matching(self, target: Any, shallow = False) -> bool:
        return type(target) is self.type

    def get_syntax_string(self, continue_=False) -> str:
        return f"{self.type.__name__}"

    def parse_value(self, target: Any, parse_blindly=False) -> Any:
        if not self.is_matching(target):
            self.raise_parse_error(target)

        return self.constructor(target)


class Enumerated(Parser):
    def __init__(
        self, values: list[Any], constructor: Optional[Callable[[Any], Any]] = None
    ) -> None:
        self.values = set(values)
        self.types = {type(x).__qualname__ for x in values}
        self.constructor = constructor or (identity)

    def is_matching(self, target: Any, shallow = False) -> bool:
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
    def __init__(
        self, value: Any, constructor: Optional[Callable[[Any], Any]] = None
    ) -> None:
        self.value = value
        self.constructor = constructor or (identity)

    def is_matching(self, target: Any, shallow = False) -> bool:
        return target == self.value

    def get_syntax_string(self, continue_=False) -> str:
        return f"const{self.value}"

    def parse_value(self, target: Any, parse_blindly=False) -> Any:
        if not self.is_matching(target):
            self.raise_parse_error(target)

        return self.constructor(target)


class Identity(Parser):
    def __init__(
        self, matcher: Parser, constructor: Optional[Callable[[Any], Any]] = None
    ) -> None:
        self.matcher = matcher
        self.constructor = constructor or (identity)

    def is_matching(self, target: Any, shallow = False) -> bool:
        return self.matcher.is_matching(target, shallow)

    def get_syntax_string(self, continue_=False) -> str:
        return self.matcher.get_syntax_string(continue_)

    def parse_value(self, target: Any, parse_blindly=False) -> Any:
        if not self.is_matching(target):
            self.raise_parse_error(target)

        return self.constructor(self.matcher.parse_value(target, parse_blindly=True))


class DictOf(Parser):
    def __init__(
        self,
        matcher: Parser,
        constructor: Optional[Callable[[dict], Any]] = None,
        key_is_allowed: Optional[Callable[[str], bool]] = None,
    ) -> None:
        self.matcher = matcher
        self.constructor = constructor or (identity)
        self.key_is_allowed = key_is_allowed or (lambda _: True)

    def is_matching(self, target: Any, shallow = False) -> bool:
        if type(target) is not dict:
            return False

        if not all(self.key_is_allowed(key) for key in target):
            return False

        return all(self.matcher.is_matching(value, shallow) for value in target.values())

    def get_syntax_string(self, continue_=False) -> str:
        return "{ [str]: " + self.matcher.get_syntax_string(continue_) + " }"

    def parse_value(self, target: Any, parse_blindly=False) -> Any:
        if type(target) is dict:
            additional_messages = [f'Unexpected key "{key}"' for key in target.keys() if not self.key_is_allowed(key)]
            if additional_messages:
                self.raise_parse_error(target, additional_messages)
        else:
            self.raise_parse_error(target)

        return self.constructor(
            {
                key: self.matcher.parse_value(value, parse_blindly)
                for key, value in target.items()
            }
        )


class ListOf(Parser):
    def __init__(
        self, matcher: Parser, constructor: Optional[Callable[[list], Any]] = None
    ) -> None:
        self.matcher = matcher
        self.constructor = constructor or (identity)

    def is_matching(self, target: Any, shallow = False) -> bool:
        if type(target) is not list:
            return False

        return all(self.matcher.is_matching(v) for v in target)

    def get_syntax_string(self, continue_=False) -> str:
        return f"{self.matcher.get_syntax_string(continue_)}[]"

    def parse_value(self, target: Any, parse_blindly=False) -> Any:
        if not parse_blindly and not self.is_matching(target):
            self.raise_parse_error(target)

        return self.constructor(
            [self.matcher.parse_value(value, parse_blindly=True) for value in target]
        )


class Opt(Parser):
    def __init__(self, matcher: Parser) -> None:
        self.matcher = matcher

    def parse_value(self, target, parse_blindly=False) -> Any:
        return self.matcher.parse_value(target, parse_blindly)

    def is_matching(self, target, shallow = False) -> bool:
        return self.matcher.is_matching(target, shallow)

    def get_syntax_string(self, continue_: bool) -> str:
        return self.matcher.get_syntax_string(continue_)


class DictExp(Parser):
    def __init__(
        self,
        types_dict: dict[Any, Union[Parser, Opt]],
        constructor: Optional[Callable[[dict], Any]] = None,
    ) -> None:
        self.types_dict = types_dict
        self.constructor = constructor or (identity)

    def is_matching(self, target, shallow = False) -> bool:
        if type(target) is not dict:
            return False

        for key in target.keys():
            if key not in self.types_dict:
                return False

        for key, matcher in self.types_dict.items():
            if type(matcher) is not Opt and key not in target:
                return False

        if not shallow:
            for key, matcher in self.types_dict.items():
                if type(matcher) is Opt:
                    if key not in target:
                        continue
                if not matcher.is_matching(target[key]):
                    return False

        return True

    def get_syntax_string(self, continue_=False) -> str:
        return indent(
            "{"
            + "".join(
                [
                    "\n  " + (
                        f"{key}: ?({matcher.matcher.get_syntax_string(continue_)})"
                        if type(matcher) is Opt
                        else f"{key}: {matcher.get_syntax_string(continue_)}"
                    )  + ","
                    for key, matcher in self.types_dict.items()
                ]
            )
            + "\n}",
            "  "
        )

    def parse_value(self, target: Any, parse_blindly=False) -> Any:
        if type(target) is dict:
            additional_messages = [f'Unexpected key "{key}"' for key in target.keys() if key not in self.types_dict]
            additional_messages.extend([
                f'Expected key "{key}"'
                for key in self.types_dict.keys()
                if (type(self.types_dict[key]) is not Opt) and (key not in target)
            ])
            if len(additional_messages) > 0:
                self.raise_parse_error(target, additional_messages)
        else:
            self.raise_parse_error(target)

        return self.constructor({
            key: (
                matcher.matcher.parse_value(target[key], parse_blindly=True)
                if type(matcher) is Opt
                else matcher.parse_value(target[key], parse_blindly=True)
            )
            for key, matcher in self.types_dict.items()
            if key in target
        })


class UnionExp(Parser):
    def __init__(
        self, *matchers: Parser, constructor: Optional[Callable[[Any], Any]] = None
    ) -> None:
        self.matchers = matchers
        self.constructor = constructor or (identity)

    def is_matching(self, target: Any, shallow = False) -> bool:
        return any(matcher.is_matching(target, shallow) for matcher in self.matchers)

    def get_syntax_string(self, continue_=False) -> str:
        return indent("".join(["\n| " + m.get_syntax_string(continue_).lstrip(" ") for m in self.matchers]), "  ")

    def parse_value(self, target: Any, parse_blindly=False) -> Any:
        messages: list[str] = []

        for matcher in self.matchers:
            if matcher.is_matching(target, True):
                try:
                    parsed = matcher.parse_value(target)
                    return self.constructor(parsed)
                except SyntaxPrasingError as err:
                    messages.append(err.args[0])
        else:
            self.raise_parse_error(target, messages)


class Scoped(Parser):
    def __init__(
        self,
        scope: "Scope",
        name: str,
        constructor: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        self.scope = scope
        self.name = name
        self.constructor = constructor or (identity)

    def is_matching(self, target: Any, shallow = False) -> bool:
        return self.scope.get_scoped_parser(self.name).is_matching(target, shallow)

    def get_syntax_string(self, continue_=False) -> str:
        if not continue_:
            return f"{self.scope.scope_name}::{self.name}"

        return indent(
            f"{self.scope.scope_name}::{self.name} = \n"
            + self.scope.get_scoped_parser(self.name).get_syntax_string(False),
            "  "
        )

    def parse_value(self, target: Any, parse_blindly=False) -> Any:
        return self.constructor(
            self.scope.get_scoped_parser(self.name).parse_value(target, parse_blindly)
        )


class Scope:
    def __init__(
        self,
        scope_name: str,
        *,
        parser_assembler: Callable[[Callable[[str], Parser]], dict[str, Parser]],
    ) -> None:
        self.scope_name = scope_name
        self.types_dict = parser_assembler(lambda name: Scoped(self, name))

    def get_scoped_parser(self, name: str) -> Parser:
        if name not in self.types_dict:
            raise ValueError(
                f"Scoped Parser construction failed, {self.scope_name}::{name} does not exist."
            )

        return self.types_dict[name]
