from inspect import cleandoc

FUNCTION_TEMPLATE = cleandoc(
    """
    namespace functions {{{description}
      type {name} = (_: {{{parameters}
      }}) => any;
    }} // namespace functions
    """
)
PARAMETER_TEMPLATE = cleandoc(
    """{description}
    {name}{optional}: {type_};
    """
)
DESCRIPTION_TEMPLATE = "\n// {description}"
