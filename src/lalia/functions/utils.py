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


def extract_enum_type(schema, property_name):
    property_info = schema.get("properties", {}).get(property_name, {})

    enum_type = None

    if "allOf" in property_info:
        for condition in property_info["allOf"]:
            if "$ref" in condition:
                ref_path = condition["$ref"].strip("#/").split("/")
                ref_schema = schema
                for part in ref_path:
                    ref_schema = ref_schema.get(part, {})
                enum_type = ref_schema.get("type")
                break  # assuming only one $ref in 'allOf' for this use case

    return enum_type


def extract_param_type(info, name, schema):
    # if its a normal parameter, it has a type key
    if "type" in info:
        return info.get("type")
    # if its a union type parameter, it has a anyOf key
    if "anyOf" in info:
        types = [d.get("type") for d in info["anyOf"] if "type" in d]
        return " | ".join(f'"{t}"' for t in types)
    if "allOf" in info:
        return extract_enum_type(schema, name)

    return None
