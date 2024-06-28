import inspect
import json
import re
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

from jsonschema.protocols import Validator
from pydantic import create_model
from referencing import Registry, Resource
from referencing._core import Resolver
from referencing.jsonschema import DRAFT202012

# allow `\"`, `\\`, or any character which isn't a control sequence
STRING_INNER = r'([^"\\\x00-\x1F\x7F-\x9F]|\\["\\])'
STRING = f'"{STRING_INNER}*"'

INTEGER = r"(-)?(0|[1-9][0-9]*)"
NUMBER = rf"({INTEGER})(\.[0-9]+)?([eE][+-][0-9]+)?"
BOOLEAN = r"(true|false)"
NULL = r"null"
WHITESPACE = r"[ ]?"

type_to_regex = {
    "string": STRING,
    "integer": INTEGER,
    "number": NUMBER,
    "boolean": BOOLEAN,
    "null": NULL,
}

DATE_TIME = r'"(-?(?:[1-9][0-9]*)?[0-9]{4})-(1[0-2]|0[1-9])-(3[01]|0[1-9]|[12][0-9])T(2[0-3]|[01][0-9]):([0-5][0-9]):([0-5][0-9])(\.[0-9]{3})?(Z)?"'
DATE = r'"(?:\d{4})-(?:0[1-9]|1[0-2])-(?:0[1-9]|[1-2][0-9]|3[0-1])"'
TIME = r'"(2[0-3]|[01][0-9]):([0-5][0-9]):([0-5][0-9])(\\.[0-9]+)?(Z)?"'
UUID = r'"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"'

format_to_regex = {
    "uuid": UUID,
    "date-time": DATE_TIME,
    "date": DATE,
    "time": TIME,
}


def build_regex_from_schema(schema: str, whitespace_pattern: Optional[str] = None):
    """Turn a JSON schema into a regex that matches any JSON object that follows
    this schema.

    JSON Schema is a declarative language that allows to annotate JSON documents
    with types and descriptions. These schemas can be generated from any Python
    datastructure that has type annotation: namedtuples, dataclasses, Pydantic
    models. And by ensuring that the generation respects the schema we ensure
    that the output can be parsed into these objects.
    This function parses the provided schema and builds a generation schedule which
    mixes deterministic generation (fixed strings), and sampling with constraints.

    Parameters
    ----------
    schema
        A string that represents a JSON Schema.
    whitespace_pattern
        Pattern to use for JSON syntactic whitespace (doesn't impact string literals)
        Example: allow only a single space or newline with `whitespace_pattern=r"[\n ]?"`

    Returns
    -------
    A generation schedule. A list of strings that represent the JSON
    schema's structure and regular expression that define the structure of
    the fields.

    References
    ----------
    .. [0] JSON Schema. https://json-schema.org/

    """

    schema = json.loads(schema)
    Validator.check_schema(schema)

    # Build reference resolver
    schema = Resource(contents=schema, specification=DRAFT202012)
    uri = schema.id() if schema.id() is not None else ""
    registry = Registry().with_resource(uri=uri, resource=schema)
    resolver = registry.resolver()

    content = schema.contents

    return JSONSchemaRegexGenerator(resolver, whitespace_pattern).to_regex(content)


def validate_quantifiers(
    min_bound: Optional[str], max_bound: Optional[str], start_offset: int = 0
) -> Tuple[str, str]:
    """
    Ensures that the bounds of a number are valid. Bounds are used as quantifiers in the regex.

    Parameters
    ----------
    min_bound
        The minimum value that the number can take.
    max_bound
        The maximum value that the number can take.
    start_offset
        Number of elements that are already present in the regex but still need to be counted.
        ex: if the regex is already "(-)?(0|[1-9][0-9])", we will always have at least 1 digit, so the start_offset is 1.

    Returns
    -------
    min_bound
        The minimum value that the number can take.
    max_bound
        The maximum value that the number can take.

    Raises
    ------
    ValueError
        If the minimum bound is greater than the maximum bound.

    TypeError or ValueError
        If the minimum bound is not an integer or None.
        or
        If the maximum bound is not an integer or None.
    """
    min_bound = "" if min_bound is None else str(int(min_bound) - start_offset)
    max_bound = "" if max_bound is None else str(int(max_bound) - start_offset)
    if min_bound and max_bound:
        if int(max_bound) < int(min_bound):
            raise ValueError("max bound must be greater than or equal to min bound")
    return min_bound, max_bound


def get_schema_from_signature(fn: Callable) -> str:
    """Turn a function signature into a JSON schema.

    Every JSON object valid to the output JSON Schema can be passed
    to `fn` using the ** unpacking syntax.

    """
    signature = inspect.signature(fn)
    arguments = {}
    for name, arg in signature.parameters.items():
        if arg.annotation == inspect._empty:
            raise ValueError("Each argument must have a type annotation")
        else:
            arguments[name] = (arg.annotation, ...)

    try:
        fn_name = fn.__name__
    except Exception as e:
        fn_name = "Arguments"
        warnings.warn(
            f"The function name could not be determined. Using default name 'Arguments' instead. For debugging, here is exact error:\n{e}",
            category=UserWarning,
        )
    model = create_model(fn_name, **arguments)

    return model.model_json_schema()


class JSONSchemaRegexGenerator:
    """Translate a JSON Schema instance into a regex that validates the schema.

    Note
    ----
    Many features of JSON schema are missing:
    - Handle `additionalProperties` keyword
    - Handle types defined as a list
    - Handle constraints on numbers
    - Handle special patterns: `date`, `uri`, etc.

    This does not support recursive definitions.

    Parameters
    ----------
    resolver
        An object that resolves references to other instances within a schema
    whitespace_pattern
        Pattern to use for JSON syntactic whitespace (doesn't impact string literals)
        Example: allow only a single space or newline with `whitespace_pattern=r"[\n ]?"`
    recursion_level
        For unconstrained objects and lists ond many levels deep the pattern should be constructed.
    """

    def __init__(
        self,
        resolver: Resolver,
        whitespace_pattern: Optional[str] = None,
        recursion_level: int = 2,
    ):
        self.resolver = resolver
        self.ws = WHITESPACE if whitespace_pattern is None else whitespace_pattern
        self.recursion_level = recursion_level
        self.types = [
            {"type": "boolean"},
            {"type": "null"},
            {"type": "number"},
            {"type": "integer"},
            {"type": "string"},
            {"type": "array"},
            {"type": "object"},
        ]

    def to_regex(self, node: dict, depth: int = 0):
        if node == {}:
            handler = self.visit_unconstrained
        elif "const" in node:
            handler = self.visit_const
        elif "allOf" in node:
            handler = self.visit_allOf
        elif "anyOf" in node:
            handler = self.visit_anyOf
        elif "oneOf" in node:
            handler = self.visit_oneOf
        elif "not" in node:
            handler = self.visit_not
        elif "$ref" in node:
            handler = self.visit_ref
        elif "enum" in node:
            handler = self.visit_enum
        elif "prefixItems" in node:
            handler = self.visit_prefixItems
        elif "properties" in node:
            handler = self.visit_object

        elif "type" in node:
            # The type keyword may either be a string or an array:
            # - If it's a string, it is the name of one of the basic types.
            # - If it is an array, it must be an array of strings, where each string is
            if isinstance(node["type"], list):
                handler = self.visit_many_types
            else:
                # Determine the type of node or default to generic handling
                handler = getattr(self, f"visit_{node['type']}", self.visit_default)
        else:
            handler = self.visit_default

        pattern = handler(node, depth + 1)
        if depth > 0:
            return rf"({pattern})"
        return pattern

    def visit_unconstrained(self, node: dict, depth: int):
        return self._create_depth_limited_unconstrained(node, depth)

    def visit_const(self, node: dict, depth: int):
        return self.format_literal(node["const"])

    def visit_boolean(self, node: dict, depth: int):
        return self.format_boolean()

    def visit_null(self, node: dict, depth: int):
        return self.format_null()

    def visit_number(self, node: dict, depth: int):
        quantifier_keys = [
            "minDigitsInteger",
            "maxDigitsInteger",
            "minDigitsFraction",
            "maxDigitsFraction",
            "minDigitsExponent",
            "maxDigitsExponent",
        ]
        if any([qk in node for qk in quantifier_keys]):
            min_digits_integer, max_digits_integer = validate_quantifiers(
                node.get("minDigitsInteger"),
                node.get("maxDigitsInteger"),
                start_offset=1,
            )
            min_digits_fraction, max_digits_fraction = validate_quantifiers(
                node.get("minDigitsFraction"), node.get("maxDigitsFraction")
            )
            min_digits_exponent, max_digits_exponent = validate_quantifiers(
                node.get("minDigitsExponent"), node.get("maxDigitsExponent")
            )
            return self.format_number_range(
                min_digits_integer,
                max_digits_integer,
                min_digits_fraction,
                max_digits_fraction,
                min_digits_exponent,
                max_digits_exponent,
            )
        else:
            return self.format_number()

    def visit_integer(self, node: dict, depth: int):
        min_digits, max_digits = validate_quantifiers(
            node.get("minDigits"), node.get("maxDigits"), start_offset=1
        )
        if min_digits is not None or max_digits is not None:
            return self.format_integer_range(min_digits, max_digits)
        else:
            return self.format_integer()

    def visit_string(self, node: dict, depth: int):
        if "maxLength" in node or "minLength" in node:
            min_length, max_length = validate_quantifiers(
                node.get("minLength"), node.get("maxLength")
            )
            return self.format_string_length(min_length, max_length)
        elif "pattern" in node:
            return self.format_string_pattern(node["pattern"])
        elif "format" in node:
            return self.format_string_format(node["format"])
        return self.format_string()

    def visit_array(self, node: dict, depth: int):
        min_items = node.get("minItems")
        max_items = node.get("maxItems")

        if "items" in node:
            items_regex = self.to_regex(node["items"], depth)
        else:
            items_regex = self._create_depth_limited_unconstrained(node, depth)

        return self.format_array(items_regex, min_items, max_items)

    def visit_object(self, node: dict, depth: int):
        """
        TODO: clean up, improve doc-string
        Handles object with no constraints, properties, or additionalProperties

        additionalProperties handling:
            pattern for json object with values defined by instance["additionalProperties"]
            enforces value type constraints recursively, "minProperties", and "maxProperties"
            doesn't enforce "required", "dependencies", "propertyNames" "any/all/on Of"

        properties handling:
            TODO
        """

        properties = node.get("properties", {})
        required_properties = node.get("required", [])
        additional_properties = node.get("additionalProperties")

        min_properties = node.get("minProperties")
        max_properties = node.get("maxProperties")

        if additional_properties:
            # handle additional propreties constraint is set
            if properties:
                raise NotImplementedError(
                    "Cannot use `properties` and `additionalProperties` simultaneously"
                )
            elif additional_properties is False:
                # Empty object if explicitly False
                return self.format_empty_object()
            else:
                if additional_properties is True:
                    # Object with unconstrained values if explicitly True
                    additional_properties_value_pattern = (
                        self._create_depth_limited_unconstrained(node, depth)
                    )
                else:
                    # Object with arbitrary key name, constrained value
                    additional_properties_value_pattern = self.to_regex(
                        additional_properties, depth
                    )
                return self.format_object_with_additional_properties(
                    additional_properties_value_pattern,
                    min_properties=min_properties,
                    max_properties=max_properties,
                )

        elif properties:
            # handle properties constraint set

            if min_properties is not None or max_properties is not None:
                raise NotImplementedError(
                    "Cannot handle object with properties and minProperties / maxProperties"
                )

            property_details = []
            for name, value in properties.items():
                property_details.append(
                    {
                        "key_pattern": f'"{self._escape_string(name)}"',
                        "value_pattern": self.to_regex(value, depth),
                        "is_required": name in required_properties,
                    }
                )
            return self.format_object_with_properties(property_details)

        else:
            # handle no property / additionalProperties constraints
            additional_properties_value_pattern = (
                self._create_depth_limited_unconstrained(node, depth)
            )
            return self.format_object_with_additional_properties(
                additional_properties_value_pattern,
                min_properties=min_properties,
                max_properties=max_properties,
            )

    def visit_enum(self, node: dict, depth: int):
        """
        The enum keyword is used to restrict a value to a fixed set of values. It
        must be an array with at least one element, where each element is unique.
        """
        choices = [self.format_literal(choice) for choice in node["enum"]]
        return self._regex_xor(choices)

    def visit_ref(self, node: dict, depth: int):
        path = node["$ref"]
        new_node = self.resolver.lookup(path).contents
        return self.to_regex(new_node, depth)

    def visit_allOf(self, node: dict, depth: int):
        subpatterns = [self.to_regex(subschema, depth) for subschema in node["allOf"]]
        return self._regex_and(subpatterns)

    def visit_anyOf(self, node: dict, depth: int):
        subpatterns = [self.to_regex(subschema, depth) for subschema in node["anyOf"]]
        # TODO: implement correctly
        # return self._regex_xor(subpatterns)
        return self._regex_xor(subpatterns)

    def visit_oneOf(self, node: dict, depth: int):
        subpatterns = [self.to_regex(subschema, depth) for subschema in node["oneOf"]]
        return self._regex_xor(subpatterns)

    def visit_not(self, node: dict, depth: int):
        raise NotImplementedError("`not` key in json schema isn't implemented")

    def visit_prefixItems(self, node: dict, depth: int):
        """
        TODO: better docstring
        Create pattern for Tuples, per JSON Schema spec, `prefixItems` determines types at each idx
        """
        subpatterns = [self.to_regex(item, depth) for item in node["prefixItems"]]
        return self.format_prefixItems(subpatterns)

    def visit_many_types(self, node: dict, depth: int):
        subpatterns = [self.to_regex({"type": t}, depth) for t in node["type"]]
        return self._regex_xor(subpatterns)

    def visit_default(self, node: dict, depth: int):
        raise NotImplementedError(f"Handler for {node['type']} is not implemented")

    def format_boolean(self):
        return type_to_regex["boolean"]

    def format_null(self):
        return type_to_regex["null"]

    def format_number(self):
        return type_to_regex["number"]

    def format_integer(self, min_digits=None, max_digits=None):
        return type_to_regex["integer"]

    def format_number_range(
        self,
        min_digits_integer,
        max_digits_integer,
        min_digits_fraction,
        max_digits_fraction,
        min_digits_exponent,
        max_digits_exponent,
    ):
        integers_quantifier = (
            f"{{{min_digits_integer},{max_digits_integer}}}"
            if min_digits_integer or max_digits_integer
            else "*"
        )
        fraction_quantifier = (
            f"{{{min_digits_fraction},{max_digits_fraction}}}"
            if min_digits_fraction or max_digits_fraction
            else "+"
        )
        exponent_quantifier = (
            f"{{{min_digits_exponent},{max_digits_exponent}}}"
            if min_digits_exponent or max_digits_exponent
            else "+"
        )
        return rf"((-)?(0|[1-9][0-9]{integers_quantifier}))(\.[0-9]{fraction_quantifier})?([eE][+-][0-9]{exponent_quantifier})?"

    def format_integer_range(self, min_digits=None, max_digits=None):
        if min_digits or max_digits:
            num_items_pattern = f"{{{min_digits},{max_digits}}}"
        else:
            num_items_pattern = "*"
        return rf"(-)?(0|[1-9][0-9]{num_items_pattern})"

    def format_string(self):
        return type_to_regex["string"]

    def format_string_length(self, min_length, max_length):
        return f'"{STRING_INNER}{{{min_length},{max_length}}}"'

    def format_string_pattern(self, pattern: str):
        return f'"{pattern[1:-1] if pattern[0] == "^" and pattern[-1] == "$" else pattern}"'

    def format_string_const(self, const: Any):
        return re.escape(json.dumps(const))

    def format_string_format(self, fmt: str):
        format_regex = format_to_regex.get(fmt)
        if format_regex:
            return format_regex
        raise NotImplementedError(f"Format {fmt} is not supported")

    def format_empty_object(self):
        return r"\{" + self.ws + r"\}"

    def format_object_with_properties(self, property_details: List[Dict]):
        properties = [
            (prop["key_pattern"], prop["value_pattern"]) for prop in property_details
        ]
        is_required = [prop["is_required"] for prop in property_details]

        # helper to construct an individual elem
        create_property_pattern = (
            lambda key_pat, value_pat: f"{self.ws}{key_pat}{self.ws}:{self.ws}{value_pat}"
        )

        inner = ""
        if any(is_required):
            last_required_pos = max([i for i, value in enumerate(is_required) if value])
            for i, (key_pattern, value_pattern) in enumerate(properties):
                subregex = create_property_pattern(key_pattern, value_pattern)
                if i < last_required_pos:
                    subregex = f"{subregex}{self.ws},"
                elif i > last_required_pos:
                    subregex = f"{self.ws},{subregex}"
                inner += subregex if is_required[i] else f"({subregex})?"
        else:
            property_subregexes = [
                create_property_pattern(key_pattern, value_pattern)
                for key_pattern, value_pattern in properties
            ]
            possible_patterns = []
            for i in range(len(property_subregexes)):
                pattern = ""
                for subregex in property_subregexes[:i]:
                    pattern += f"({subregex}{self.ws},)?"
                pattern += property_subregexes[i]
                for subregex in property_subregexes[i + 1 :]:
                    pattern += f"({self.ws},{subregex})?"
                possible_patterns.append(pattern)
            inner += f"({'|'.join(possible_patterns)})?"

        return r"\{" + inner + self.ws + r"\}"

    def format_object_with_additional_properties(
        self, value_pattern: str, min_properties=None, max_properties=None
    ):
        inner = self._regex_repeat_elem(
            elem_pattern=f"({STRING}){self.ws}:{self.ws}({value_pattern})",
            separator_pattern=f"{self.ws},{self.ws}",
            min_elem=min_properties,
            max_elem=max_properties,
            pad=self.ws,
        )
        return r"\{" + inner + r"\}"

    def format_array(self, elem_pattern: str, min_items=None, max_items=None):
        inner = self._regex_repeat_elem(
            elem_pattern=elem_pattern,
            separator_pattern=f"{self.ws},{self.ws}",
            min_elem=min_items,
            max_elem=max_items,
            pad=self.ws,
        )
        return rf"\[{inner}\]"

    def format_prefixItems(self, subpatterns: List[str]):
        # PR TODO: this doesn't seem to be implemented correctly
        comma_split_pattern = rf"{self.ws},{self.ws}"
        return rf"\[{self.ws}{comma_split_pattern.join(subpatterns)}{self.ws}\]"

    def format_literal(self, literal: Any):
        if type(literal) in [int, float, bool, type(None), str]:
            return re.escape(json.dumps(literal))
        else:
            raise TypeError(f"Unsupported data type in literal: {type(literal)}")

    def _regex_and(self, patterns: List[str]):
        return "".join(patterns)

    def _regex_xor(self, patterns: List[str]):
        pat = "|".join(patterns)
        return f"{(pat)}"

    def _regex_or(self, patterns: List[str]):
        raise NotImplementedError

    def _regex_repeat_elem(
        self,
        elem_pattern: str,
        separator_pattern: str,
        min_elem=None,
        max_elem=None,
        pad="",
    ):
        """
        Creates a pattern allowing between min_elem and max_elem occurrences of elem_pattern
        Ensures each element pattern is separated by separator_pattern
        Surrounds result with `pad`
        """
        if str(max_elem) == "0":
            return pad

        base_pattern = f"({elem_pattern})"
        suffix_pattern = f"(({separator_pattern})({elem_pattern}))"

        min_suffix_repeats = "" if min_elem is None else max(0, int(min_elem) - 1)
        max_suffix_repeats = "" if max_elem is None else max_elem - 1

        if str(max_suffix_repeats) == "0":
            pattern = base_pattern
        else:
            pattern = f"{base_pattern}({suffix_pattern}){{{min_suffix_repeats},{max_suffix_repeats}}}"

        padded_pattern = f"({pad}{pattern}{pad})"

        if not min_elem:
            return f"({padded_pattern}|{pad})"
        else:
            return padded_pattern

    def _escape_string(self, string: str):
        return re.escape(string)

    def _create_depth_limited_unconstrained(self, node: dict, depth: int):
        legal_types = [
            {"type": "boolean"},
            {"type": "null"},
            {"type": "number"},
            {"type": "integer"},
            {"type": "string"},
        ]
        allowed_depth = node.get("_allowed_depth", depth + self.recursion_level)
        # We limit the object depth to keep the expression finite, but the "depth"
        # key is not a true component of the JSON Schema specification.
        if depth < allowed_depth:
            legal_types.append({"type": "object", "_allowed_depth": allowed_depth})
            legal_types.append({"type": "array", "_allowed_depth": allowed_depth})
        return self.visit_oneOf({"oneOf": legal_types}, depth)
