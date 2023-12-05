from types import BuiltinFunctionType, FunctionType


def is_callable_instance(callable_: object) -> bool:
    if not callable(callable_):
        return False
    return not isinstance(callable_, FunctionType | BuiltinFunctionType)


def extract_enum(schema, property_name):
    enum_vals = []
    if property_name in schema.get("properties", {}):
        property_details = schema["properties"][property_name]

        if "allOf" in property_details:
            for ref in property_details["allOf"]:
                if "$ref" in ref:
                    ref_path = ref["$ref"].strip("#/").split("/")
                    ref_schema = schema
                    for part in ref_path:
                        ref_schema = ref_schema.get(part, {})
                    if "enum" in ref_schema:
                        enum_vals = ref_schema["enum"]

    if enum_vals:
        return enum_vals
    else:
        return None
