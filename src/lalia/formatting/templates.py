from inspect import cleandoc

NAMESPACE_TEMPLATE = cleandoc(
    """// Tools

    // Functions

    namespace functions {{
    {functions}

    }} // namespace functions
    """
)
FUNCTION_TEMPLATE = cleandoc(
    """{description}
    type {name} = {parameters} => any;
    """
)
PARAMETER_TEMPLATE = cleandoc("""{description}{name}{optional}: {type_},{default}""")
