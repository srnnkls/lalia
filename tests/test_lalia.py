import pprint
import time
from importlib.resources import files
from inspect import cleandoc
from typing import Annotated, Any

import psycopg
import sqlparse
from langchain.callbacks.openai_info import MODEL_COST_PER_1K_TOKENS
from ruamel.yaml import YAML
from sqloxide import mutate_relations, parse_sql  # type: ignore

from cobi import assets
from cobi.utils.auth.env import Postgres
from cobi.utils.auth.secrets import get_openai_token
from cobi.utils.db.shopware import (
    PG_KEYWORDS,
    extract_columns,
    extract_unified_all_foreign_keys,
    find_paths_both_directions,
    to_dinsql,
    to_fdw_view_representation,
)
from lalia.chat.messages import AssistantMessage
from lalia.chat.session import Session
from lalia.llm.openai import ChatModel, OpenAIChat

yaml = YAML(typ="string")
yaml.indent(mapping=2, sequence=4, offset=2)

MD_YAML_TEMPLATE = cleandoc(
    """
    ```yaml
    {data}
    ```
    """
)


SCHEMA_FILE = files(assets) / "db/shopware/schema.yml"
POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB = Postgres.from_env_file.values()

DB_URI = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@localhost/{POSTGRES_DB}"

with open(SCHEMA_FILE) as f:  # type: ignore
    schema = yaml.load(f)

schema_fdw = to_fdw_view_representation(schema)

tables = extract_columns(schema_fdw)
unified_foreign_keys = extract_unified_all_foreign_keys(schema_fdw)

schema_din_sql = "\n".join(
    line for group in to_dinsql(tables, unified_foreign_keys) for line in group
)

conn = psycopg.connect(DB_URI)


def sql_db_pre_select_tables(
    join_path: Annotated[
        list[str],
        cleandoc(
            """
            A list cotaining the join path with all tables in the right order for
            the query and returns the schema for a given set of tables.
            """
        ),
    ]
) -> str:
    """
    Receives a list cotaining the join path with all tables in the right order for
    the query and returns the schema for a given set of tables.
    """
    join_path = [table.strip('"') for table in join_path]
    origin, *_, target = join_path if len(join_path) > 2 else join_path * 2

    all_paths = find_paths_both_directions(origin, target, schema_fdw)

    # if set(join_path) not in [set(path) for path in all_paths]:
    if join_path not in all_paths and list(reversed(join_path)) not in all_paths:
        return (
            "Error: The given join_path is not valid. "
            "Remember to put tables in right order!"
        )

    columns = extract_columns(
        schema_fdw,
        tables=join_path,
    )
    unified_foreign_keys = extract_unified_all_foreign_keys(
        schema_fdw,
        tables=join_path,
    )

    return "\n".join(
        line for group in to_dinsql(columns, unified_foreign_keys) for line in group
    )


def sql_db_check_query(query: Annotated[str, "The SQL query to check."]) -> str:
    """
    Use this function to double check if your query is correct before executing it.

    Use this function before executing a query with sql_db_execute_query! Use the
    reproduced query as input for sql_db_execute_query.
    """
    template = cleandoc(
        """
        The query looks good, here is the reproduced query:
        ```sql
        {formatted_query}
        ```
        """
    )

    try:
        parsed_query = parse_sql(query, dialect="postgres")
    except ValueError as e:
        return f"Error: Syntax error. {e}"
    else:

        def escape_keywords(relation: str) -> str:
            if relation.upper() in PG_KEYWORDS:
                return f'"{relation}"'
            return relation

        corrected = ";\n\n\n".join(
            mutate_relations(parsed_query=parsed_query, func=escape_keywords)
        )
    formatted_query = sqlparse.format(corrected, reindent=True, keyword_case="upper")
    return template.format(formatted_query=formatted_query)


def sql_db_execute_query(query: Annotated[str, "The SQL query to execute."]) -> str:
    """
    Function to query Postgres database with a provided SQL query. The SQL dialect is
    PostgreSQL.

    If an error is returned, rewrite the query, check the query, and try
    again.

    Always use the reproduced query from the sql_db_check_query function as input.
    """
    try:
        with psycopg.connect(DB_URI) as conn:
            return str(conn.execute(query).fetchall())  # type: ignore
    except psycopg.Error as e:
        return f"Error: Query failed with {e}. Please adjust the query and try again."


SYSTEM_MESSAGE = cleandoc(
    """
    You are an agent designed to interact with a SQL database. Given an input question,
    create a syntactically correct PostgresSQL query to run, then look at the results of
    the query and return the answer.  Unless the user specifies a specific number of
    examples they wish to obtain, never limit your queries.  You can order the results
    by a relevant column to return the most interesting examples in the database.  Never
    query for all the columns from a specific table, only ask for the relevant columns
    given the question.

    ALWAYS check your queries using the sql_db_check_query before executing them.

    IF you receive an error look at the schema and try to adjust the query.

    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

    If the question does not seem related to the database, just return "I don't know" as
    the answer.

    The database schema:
    {schema}
    """
)

session = Session(
    llm=OpenAIChat(
        model=ChatModel.GPT3_5_TURBO_0613,
        temperature=0.0,
        api_key=get_openai_token(),
        debug=True,
    ),
    system_message=SYSTEM_MESSAGE.format(schema=schema_din_sql),
    messages=[
        AssistantMessage(
            content=(
                "I should first use the sql_db_pre_select_tables "
                "function to reduce the number of tables."
            )
        ),
    ],
    functions=[sql_db_pre_select_tables, sql_db_check_query, sql_db_execute_query],
    debug=True,
)


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"Time: {(end_time - start_time):.2f} seconds")

    return wrapper


def get_tokens_used_for_last_call(responses: list[dict[str, Any]]) -> int:
    tokens_used = responses[-1]["usage"]["total_tokens"]
    for response in reversed(responses[:-1]):
        if response["choices"][0]["finish_reason"] == "stop":
            break
        tokens_used += response["usage"]["total_tokens"]
    return tokens_used


@timeit
def run_query(input_: str):
    result = session(input_)
    total_tokens = get_tokens_used_for_last_call(session.llm._responses)
    print(result.content)
    print(
        f"Total costs: {total_tokens / 1000 * MODEL_COST_PER_1K_TOKENS[session.llm.model]}"
    )


run_query("List the names of all products ordered by Tony Fischer.")
run_query("Can you include the product's price?")
run_query("List the 10 most expensive products.")
run_query("Where does Annie Toy live?")
run_query("Please give me the street and the city.")
