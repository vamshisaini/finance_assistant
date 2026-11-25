import ast
import builtins
import heapq
import os
import symtable
from collections import defaultdict
from pathlib import Path

import nbformat
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
os.chdir(PROJECT_DIR)

# Load the project-level .env before the notebook initializes its LLM.
load_dotenv(dotenv_path=PROJECT_DIR / ".env", override=True)

st.set_page_config(
    page_title="AI Personal Finance Assistant",
    page_icon="💰",
    layout="wide"
)


# ============================================================
# 2. NOTEBOOK DISCOVERY AND DEPENDENCY RESOLUTION
# ============================================================

def _assigned_names(node):
    """Return names created by a top-level notebook statement."""

    names = set()

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        names.add(node.name)

    elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            names.update(
                item.id
                for item in ast.walk(target)
                if isinstance(item, ast.Name)
                and isinstance(item.ctx, ast.Store)
            )

    elif isinstance(node, (ast.Import, ast.ImportFrom)):
        for alias in node.names:
            if alias.name == "*":
                continue
            if alias.asname:
                names.add(alias.asname)
            elif isinstance(node, ast.Import):
                names.add(alias.name.split(".")[0])
            else:
                names.add(alias.name)

    return names


def _global_dependencies(node):
    """
    Find names referenced globally by a statement.

    symtable understands function/class scopes, so local variables
    inside a function are not mistaken for notebook-level dependencies.
    """

    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)

    source = ast.unparse(module)
    table = symtable.symtable(source, "<notebook statement>", "exec")
    dependencies = set()

    def visit(scope):
        for symbol in scope.get_symbols():
            if symbol.is_global() and symbol.is_referenced():
                dependencies.add(symbol.get_name())

        for child in scope.get_children():
            visit(child)

    visit(table)
    dependencies.difference_update(_assigned_names(node))
    return dependencies


def _is_workflow_builder_call(node):
    """Recognize LangGraph workflow registration statements."""

    if not isinstance(node, ast.Expr):
        return False

    call = node.value
    if not isinstance(call, ast.Call):
        return False

    if not isinstance(call.func, ast.Attribute):
        return False

    return call.func.attr in {"add_node", "add_edge"}


def _is_state_graph_assignment(node):
    """Identify the variable receiving StateGraph(...)."""

    if not isinstance(node, ast.Assign):
        return None

    value = node.value
    if not isinstance(value, ast.Call):
        return None

    function = value.func
    is_state_graph = (
        isinstance(function, ast.Name)
        and function.id == "StateGraph"
    )

    if not is_state_graph:
        return None

    target_names = _assigned_names(node)
    return next(iter(target_names), None)


def _is_graph_compile_assignment(node, graph_variable):
    """Identify the assignment that compiles a discovered graph."""

    if not isinstance(node, (ast.Assign, ast.AnnAssign)):
        return False

    value = node.value
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "compile"
        and isinstance(value.func.value, ast.Name)
        and value.func.value.id == graph_variable
    )


def discover_notebook_implementation(notebook):
    """
    Discover the implementation by symbols and dependencies.

    No notebook cell numbers or fixed cell ranges are used.
    Notebook cell order is only a tie-breaker for independent statements.
    """

    records = []

    for cell_number, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue

        source = cell.source.strip()
        if not source:
            continue

        try:
            tree = ast.parse(source)
        except SyntaxError:
            # Unrelated malformed/demo cells are not executed.
            continue

        for statement_number, node in enumerate(tree.body):
            # Keep only definitions, imports and assignments.
            # The graph's add_node/add_edge calls are handled separately.
            supported = isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef,
                    ast.Import,
                    ast.ImportFrom,
                    ast.Assign,
                    ast.AnnAssign,
                    ast.NamedExpr
                )
            )

            is_builder = _is_workflow_builder_call(node)

            if not supported and not is_builder:
                continue

            try:
                dependencies = _global_dependencies(node)
            except Exception as exc:
                raise RuntimeError(
                    "Could not inspect a notebook implementation "
                    f"statement near notebook cell {cell_number + 1}: {exc}"
                ) from exc

            records.append({
                "cell_order": cell_number,
                "statement_order": statement_number,
                "node": node,
                "defined": _assigned_names(node),
                "dependencies": dependencies,
                "is_builder": is_builder
            })

    providers = defaultdict(list)
    graph_variables = set()

    for record in records:
        for name in record["defined"]:
            providers[name].append(record)

        graph_variable = _is_state_graph_assignment(record["node"])
        if graph_variable:
            graph_variables.add(graph_variable)

    # A valid five-agent LangGraph needs a StateGraph object, its
    # registration calls, and its compile assignment.
    if not graph_variables:
        raise RuntimeError(
            "Could not discover a LangGraph StateGraph(...) assignment "
            "in the notebook."
        )

    builder_records = [
        record for record in records
        if record["is_builder"]
        and isinstance(record["node"].value.func.value, ast.Name)
        and record["node"].value.func.value.id in graph_variables
    ]

    if not builder_records:
        raise RuntimeError(
            "The notebook contains a StateGraph object but no discoverable "
            "add_node/add_edge registration calls."
        )

    required_symbols = {
        "financial_assistant_graph",
        "FinancialAssistantState",
        "validate_transactions",
        "validate_financial_goals",
        "validate_investment_platforms",
        "calculate_monthly_summary",
        "calculate_combined_affordability",
        "financial_goals",
        "VALID_TRANSACTION_TYPES",
        "VALID_CATEGORIES",
        "CURRENT_EMERGENCY_FUND",
        "EMERGENCY_FUND_BUILD_MONTHS"
    }

    externally_supplied = {
        "__name__",
        "investment_platforms_df"
    }

    builtin_names = set(dir(builtins))
    selected_ids = set()
    selected_symbols = set()
    visiting = set()
    unresolved = set()

    def select_symbol(symbol):
        if (
            symbol in externally_supplied
            or symbol in builtin_names
            or symbol in selected_symbols
        ):
            return

        if symbol in visiting:
            return

        candidates = providers.get(symbol, [])
        if not candidates:
            unresolved.add(symbol)
            return

        # The last definition is the notebook's effective definition.
        provider = candidates[-1]
        provider_id = id(provider)

        visiting.add(symbol)
        selected_ids.add(provider_id)
        selected_symbols.update(provider["defined"])

        for dependency in provider["dependencies"]:
            select_symbol(dependency)

        # If this symbol is a StateGraph object, include all of its
        # add_node/add_edge calls, wherever they appear in the notebook.
        if symbol in graph_variables:
            for builder in builder_records:
                builder_id = id(builder)
                if builder_id not in selected_ids:
                    selected_ids.add(builder_id)

                for dependency in builder["dependencies"]:
                    select_symbol(dependency)

        visiting.remove(symbol)

    for symbol in required_symbols:
        select_symbol(symbol)

    if unresolved:
        raise RuntimeError(
            "The notebook implementation is missing required definitions "
            "or dependencies: "
            + ", ".join(sorted(unresolved))
        )

    selected = [
        record for record in records
        if id(record) in selected_ids
    ]

    # Build dependency edges between selected statements.
    record_by_id = {id(record): record for record in selected}
    outgoing = {id(record): set() for record in selected}
    indegree = {id(record): 0 for record in selected}

    for record in selected:
        current_id = id(record)

        for dependency in record["dependencies"]:
            candidates = providers.get(dependency, [])
            if not candidates:
                continue

            dependency_provider = candidates[-1]
            dependency_id = id(dependency_provider)

            if (
                dependency_id in record_by_id
                and dependency_id != current_id
                and current_id not in outgoing[dependency_id]
            ):
                outgoing[dependency_id].add(current_id)
                indegree[current_id] += 1

    # Ensure graph registration happens before workflow.compile().
    for graph_variable in graph_variables:
        graph_builders = [
            record for record in selected
            if record["is_builder"]
            and isinstance(record["node"].value.func.value, ast.Name)
            and record["node"].value.func.value.id == graph_variable
        ]

        # Register all nodes before adding graph edges. This makes the
        # graph construction independent of where those calls sit in
        # the notebook.
        add_node_records = [
            record for record in graph_builders
            if record["node"].value.func.attr == "add_node"
        ]
        add_edge_records = [
            record for record in graph_builders
            if record["node"].value.func.attr == "add_edge"
        ]

        for add_node_record in add_node_records:
            for add_edge_record in add_edge_records:
                node_id = id(add_node_record)
                edge_id = id(add_edge_record)
                if (
                    node_id != edge_id
                    and edge_id not in outgoing[node_id]
                ):
                    outgoing[node_id].add(edge_id)
                    indegree[edge_id] += 1

        compile_records = [
            record for record in selected
            if _is_graph_compile_assignment(record["node"], graph_variable)
        ]

        for compile_record in compile_records:
            compile_id = id(compile_record)

            for builder in graph_builders:
                builder_id = id(builder)
                if (
                    builder_id != compile_id
                    and compile_id not in outgoing[builder_id]
                ):
                    outgoing[builder_id].add(compile_id)
                    indegree[compile_id] += 1

    def sort_key(record):
        return (record["cell_order"], record["statement_order"])

    ready = []
    for record in selected:
        record_id = id(record)
        if indegree[record_id] == 0:
            heapq.heappush(
                ready,
                (sort_key(record), record_id)
            )

    ordered = []

    while ready:
        _, record_id = heapq.heappop(ready)
        record = record_by_id[record_id]
        ordered.append(record)

        for dependent_id in outgoing[record_id]:
            indegree[dependent_id] -= 1

            if indegree[dependent_id] == 0:
                dependent = record_by_id[dependent_id]
                heapq.heappush(
                    ready,
                    (sort_key(dependent), dependent_id)
                )

    if len(ordered) != len(selected):
        raise RuntimeError(
            "A circular or unresolved dependency was found while "
            "preparing the notebook implementation."
        )

    return ordered


def execute_notebook_implementation(notebook, namespace):
    """Execute only the dependency-resolved implementation statements."""

    ordered_statements = discover_notebook_implementation(notebook)

    # Execute imports first. This also makes the loader robust when an
    # import cell has been moved below a function or configuration cell.
    imports = [
        record for record in ordered_statements
        if isinstance(record["node"], (ast.Import, ast.ImportFrom))
    ]

    other_statements = [
        record for record in ordered_statements
        if not isinstance(record["node"], (ast.Import, ast.ImportFrom))
    ]

    for record in imports + other_statements:
        node = record["node"]
        module = ast.Module(body=[node], type_ignores=[])
        ast.fix_missing_locations(module)

        code = compile(
            module,
            (
                "<notebook implementation "
                f"cell {record['cell_order'] + 1}>"
            ),
            "exec"
        )

        exec(code, namespace)

        # Fail early with a useful message if the configured provider key
        # is absent. Never display the key itself.
        for name in record["defined"]:
            if name.endswith("_API_KEY"):
                if not namespace.get(name):
                    raise RuntimeError(
                        f"{name} is missing. Add {name}=... to the "
                        "project .env file."
                    )

    return namespace


# ============================================================
# 3. LOAD AND VALIDATE THE FINANCIAL ASSISTANT
# ============================================================

def find_notebook_path():
    candidates = [
        PROJECT_DIR / "Finance_Assistant.ipynb",
        PROJECT_DIR / "AI_Personal_Finance_Assistant.ipynb",
        PROJECT_DIR / "Finance_Assistant(2).ipynb"
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "Could not find the finance notebook. Keep "
        "Finance_Assistant.ipynb in the same folder as app.py."
    )


@st.cache_resource(show_spinner="Preparing the five-agent assistant...")
def load_financial_engine(
    notebook_path_string,
    notebook_modified_time,
    env_modified_time
):
    notebook_path = Path(notebook_path_string)

    notebook = nbformat.read(
        notebook_path,
        as_version=4
    )

    namespace = {
        "__name__": "finance_notebook_runtime"
    }

    execute_notebook_implementation(
        notebook,
        namespace
    )

    # The research agent reads this shared global at invocation time.
    platform_csv = PROJECT_DIR / "data" / "investment_platforms.csv"

    if not platform_csv.is_file():
        raise FileNotFoundError(
            "Missing data/investment_platforms.csv. "
            "Keep the investment platform CSV inside the data folder."
        )

    platforms_df = pd.read_csv(platform_csv)

    namespace["investment_platforms_df"] = (
        namespace["validate_investment_platforms"](platforms_df)
    )

    # Validate the five-agent graph and the state/report contract.
    required_runtime_symbols = {
        "financial_assistant_graph",
        "FinancialAssistantState",
        "calculate_combined_affordability",
        "financial_report_agent",
        "build_financial_report"
    }

    missing_runtime_symbols = (
        required_runtime_symbols - set(namespace)
    )

    if missing_runtime_symbols:
        raise RuntimeError(
            "Notebook initialization did not create: "
            + ", ".join(sorted(missing_runtime_symbols))
        )

    required_state_fields = {
        "transaction_df",
        "financial_goals",
        "current_emergency_fund",
        "emergency_fund_build_months",
        "investment_query",
        "financial_summary",
        "category_summary",
        "planning_results",
        "emergency_fund_plan",
        "combined_affordability",
        "financial_report_markdown"
    }

    state_fields = set(
        namespace["FinancialAssistantState"].__annotations__
    )

    missing_state_fields = required_state_fields - state_fields

    if missing_state_fields:
        raise RuntimeError(
            "FinancialAssistantState is missing required fields: "
            + ", ".join(sorted(missing_state_fields))
        )

    expected_agents = {
        "financial_analysis",
        "financial_planning",
        "investment_research",
        "risk_assessment",
        "financial_report"
    }

    graph_nodes = set(
        namespace["financial_assistant_graph"].get_graph().nodes
    )

    missing_agents = expected_agents - graph_nodes

    if missing_agents:
        raise RuntimeError(
            "The LangGraph workflow is missing agent nodes: "
            + ", ".join(sorted(missing_agents))
        )

    return namespace


# ============================================================
# 4. APPLICATION HEADER
# ============================================================

st.title(
    "💰 AI Personal Finance and Investment Assistant"
)

st.caption(
    "An India-focused, five-agent financial decision-support "
    "assistant. For educational purposes only—not professional "
    "financial advice."
)


# Initialize the five-agent system.

try:

    notebook_path = find_notebook_path()
    env_path = PROJECT_DIR / ".env"
    env_modified_time = (
        env_path.stat().st_mtime_ns if env_path.is_file() else 0
    )
    engine = load_financial_engine(
        str(notebook_path),
        notebook_path.stat().st_mtime_ns,
        env_modified_time
    )

except Exception as exc:

    st.error(
        f"Could not initialize the financial assistant: {exc}"
    )

    st.info(
        "Check that the finance notebook and data/investment_platforms.csv "
        "are in the project folder, dependencies are installed, "
        "and the API key required by the notebook is configured "
        "in .env (the uploaded notebook currently uses GROQ_API_KEY)."
    )

    st.stop()


# ============================================================
# 5. SIDEBAR INPUTS
# ============================================================

with st.sidebar:

    st.header("Financial Inputs")

    input_method = st.radio(
        "How would you like to provide transactions?",
        [
            "Upload CSV",
            "Enter manually"
        ]
    )

    st.divider()

    st.subheader("Emergency Fund")

    current_emergency_fund = st.number_input(
        "Current emergency fund (₹)",
        min_value=0.0,
        value=float(
            engine["CURRENT_EMERGENCY_FUND"]
        ),
        step=1000.0
    )

    emergency_fund_build_months = st.number_input(
        "Months to build the emergency fund",
        min_value=1,
        max_value=120,
        value=int(
            engine["EMERGENCY_FUND_BUILD_MONTHS"]
        ),
        step=1
    )

    st.divider()

    investment_query = st.text_area(
        "Investment research question",
        value=(
            "Explain the major investment options available "
            "in India, including fixed deposits, recurring "
            "deposits, mutual funds, ETFs, stocks, government "
            "securities, PPF and gold. Also explain the "
            "available investment platforms."
        ),
        height=150
    )


# ============================================================
# 6. TRANSACTION INPUT
# ============================================================

st.subheader("1. Transaction Information")


if input_method == "Upload CSV":

    st.write(
        "Upload a CSV containing the following transaction columns."
    )

    st.code(
        "transaction_id, date, description, amount, "
        "transaction_type, category, payment_mode",
        language="text"
    )

    uploaded_file = st.file_uploader(
        "Choose transaction CSV",
        type=["csv"]
    )

    transaction_df = None

    if uploaded_file is not None:

        try:

            transaction_df = pd.read_csv(uploaded_file)

            st.dataframe(
                transaction_df.head(10),
                use_container_width=True
            )

        except Exception as exc:

            st.error(
                f"Unable to read the CSV: {exc}"
            )

else:

    st.write(
        "Add your transactions below. "
        "Use the + button to add rows."
    )

    
    blank_transactions = pd.DataFrame({
        "transaction_id": pd.Series(dtype="str"),
        "date": pd.Series(dtype="datetime64[ns]"),
        "description": pd.Series(dtype="str"),
        "amount": pd.Series(dtype="float"),
        "transaction_type": pd.Series(dtype="str"),
        "category": pd.Series(dtype="str"),
        "payment_mode": pd.Series(dtype="str")
    })

    
    transaction_df = st.data_editor(
        blank_transactions,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "date": st.column_config.DateColumn(
                "date",
                format="YYYY-MM-DD",
                required=True
            ),
            "transaction_type": st.column_config.SelectboxColumn(
                "transaction_type",
                options=engine["VALID_TRANSACTION_TYPES"],
                required=True
            ),
            "category": st.column_config.SelectboxColumn(
                "category",
                options=engine["VALID_CATEGORIES"],
                required=False
            )
        },
        key="manual_transactions_editor"
    )


    transaction_df = transaction_df[
        transaction_df["transaction_id"]
        .fillna("")
        .astype(str)
        .str.strip() != ""
    ]




# ============================================================
# 7. FINANCIAL GOALS INPUT
# ============================================================

st.subheader("2. Financial Goals")

initial_goals = pd.DataFrame(
    engine["financial_goals"]
)

goals_df = st.data_editor(

    initial_goals,

    num_rows="dynamic",

    use_container_width=True,

    column_config={

        "priority": (
            st.column_config.SelectboxColumn(
                "priority",
                options=["high", "medium", "low"],
                required=True
            )
        ),

        "target_amount": (
            st.column_config.NumberColumn(
                "target_amount",
                min_value=1.0,
                step=1000.0
            )
        ),

        "current_amount": (
            st.column_config.NumberColumn(
                "current_amount",
                min_value=0.0,
                step=1000.0
            )
        ),

        "months_to_achieve": (
            st.column_config.NumberColumn(
                "months_to_achieve",
                min_value=1,
                step=1
            )
        )

    },

    key="financial_goals_editor"
)


# ============================================================
# 8. EXECUTE FINANCIAL ASSISTANT
# ============================================================

run_assistant = st.button(

    "Run Financial Assistant",

    type="primary",

    use_container_width=True

)


if run_assistant:

    try:

        # Validate transaction input.

        if transaction_df is None or transaction_df.empty:

            raise ValueError(
                "Please upload or enter at least one transaction."
            )

        validated_transactions = (
            engine["validate_transactions"](
                transaction_df
            )
        )

        # Prepare financial goals.

        clean_goals_df = goals_df.copy()

        if not clean_goals_df.empty:

            clean_goals_df = clean_goals_df[
                clean_goals_df["goal_name"]
                .fillna("")
                .astype(str)
                .str.strip() != ""
            ]

        goal_records = clean_goals_df.to_dict(
            orient="records"
        )

        # Convert month values into integers.

        for goal in goal_records:

            months = goal.get("months_to_achieve")

            if months is not None and pd.notna(months):

                if float(months).is_integer():

                    goal["months_to_achieve"] = int(months)

        financial_goals = (
            engine["validate_financial_goals"](
                goal_records
            )
        )

        if not investment_query.strip():

            raise ValueError(
                "Please enter an investment research question."
            )

        # Prepare LangGraph initial state.

        initial_state = {

            "transaction_df": validated_transactions,

            "financial_goals": financial_goals,

            "current_emergency_fund": float(
                current_emergency_fund
            ),

            "emergency_fund_build_months": int(
                emergency_fund_build_months
            ),

            "investment_query": investment_query.strip()

        }

        # Execute all five agents through LangGraph.

        with st.spinner(
            "The five agents are analyzing your information..."
        ):

            result = engine[
                "financial_assistant_graph"
            ].invoke(initial_state)

        # Store results in Streamlit session state.

        st.session_state[
            "financial_assistant_result"
        ] = result

        st.session_state[
            "financial_assistant_transactions"
        ] = validated_transactions

        st.success(
            "Financial analysis completed successfully."
        )

    except Exception as exc:

        st.error(
            f"Unable to complete the analysis: {exc}"
        )


# ============================================================
# HELPER: DISPLAY STRUCTURED AGENT OUTPUTS
# ============================================================

def display_agent_output(data):

    # Convert Pydantic models into dictionaries
    if hasattr(data, "model_dump"):
        data = data.model_dump()

    if isinstance(data, dict):

        for key, value in data.items():

            # Convert field names into readable headings
            heading = key.replace("_", " ").title()

            if isinstance(value, (dict, list)):

                st.markdown(f"**{heading}**")
                display_agent_output(value)

            elif value is None:

                continue

            else:

                st.markdown(
                    f"**{heading}:** {value}"
                )

    elif isinstance(data, list):

        for item in data:

            if isinstance(item, (dict, list)):
                display_agent_output(item)
            else:
                st.markdown(f"- {item}")

    else:

        st.write(data)

# ============================================================
# 9. DISPLAY FINANCIAL RESULTS
# ============================================================

if "financial_assistant_result" in st.session_state:

    result = st.session_state[
        "financial_assistant_result"
    ]

    transactions = st.session_state[
        "financial_assistant_transactions"
    ]

    st.divider()

    st.header("Your Financial Overview")

    # --------------------------------------------------------
    # Financial summary metrics
    # --------------------------------------------------------

    summary = result["financial_summary"]

    metric_columns = st.columns(4)

    metric_columns[0].metric(
        "Total Income",
        f"₹{summary['total_income']:,.2f}"
    )

    metric_columns[1].metric(
        "Total Expenses",
        f"₹{summary['total_expenses']:,.2f}"
    )

    metric_columns[2].metric(
        "Investments",
        f"₹{summary['total_investments']:,.2f}"
    )

    metric_columns[3].metric(
        "Cash Surplus",
        f"₹{summary['cash_surplus']:,.2f}"
    )


    # --------------------------------------------------------
    # Result tabs
    # --------------------------------------------------------

    tabs = st.tabs([

        "Overview",

        "Financial Planning",

        "Investment Research",

        "Risk Assessment",

        "Full Report"

    ])


    # ========================================================
    # TAB 1: FINANCIAL OVERVIEW
    # ========================================================

    with tabs[0]:

        left, right = st.columns(2)

        with left:

            st.subheader("Expense Categories")

            category_summary = result["category_summary"]

            if not category_summary.empty:

                st.dataframe(
                    category_summary,
                    use_container_width=True,
                    hide_index=True
                )

                category_chart = px.bar(

                    category_summary,

                    x="category",

                    y="total_amount",

                    title="Recorded Expenses by Category",

                    labels={
                        "category": "Category",
                        "total_amount": "Amount (₹)"
                    }

                )

                st.plotly_chart(
                    category_chart,
                    use_container_width=True
                )

            else:

                st.info(
                    "No expense transactions were recorded."
                )


        with right:

            st.subheader("Monthly Cash Flow")

            monthly_summary = (
                engine["calculate_monthly_summary"](
                    transactions
                )
            )

            st.dataframe(
                monthly_summary,
                use_container_width=True,
                hide_index=True
            )

            if not monthly_summary.empty:

                monthly_chart_data = monthly_summary.copy()

                monthly_chart_data["month"] = (
                    monthly_chart_data["month"].astype(str)
                )

                monthly_chart = px.line(

                    monthly_chart_data,

                    x="month",

                    y="cash_surplus",

                    markers=True,

                    title="Recorded Monthly Cash Surplus",

                    labels={
                        "month": "Month",
                        "cash_surplus": "Cash Surplus (₹)"
                    }

                )

                st.plotly_chart(
                    monthly_chart,
                    use_container_width=True
                )


        # Financial Analysis Agent
        
        st.subheader("Financial Analysis Agent")

        analysis = result["financial_analysis_result"]

        st.markdown("### Financial Health Overview")
        st.write(analysis.financial_health_overview)

        st.markdown("### Strengths")
        for item in analysis.strengths:
            st.markdown(f"- {item}")

        st.markdown("### Areas to Review")
        for item in analysis.areas_to_review:
            st.markdown(f"- {item}")

        st.markdown("### Spending Observations")
        for item in analysis.spending_observations:
            st.markdown(f"- {item}")

        st.markdown("### Data Limitations")
        for item in analysis.data_limitations:
            st.markdown(f"- {item}")

        
        st.markdown("### Overall Assessment")

        assessment = analysis.overall_assessment

        if assessment == "positive":
            st.success("Positive")

        elif assessment == "mixed":
            st.info("Mixed")

        elif assessment == "needs_attention":
            st.warning("Needs Attention")

        else:
            st.warning("Insufficient Data")




    # ========================================================
    # TAB 2: FINANCIAL PLANNING
    # ========================================================

    with tabs[1]:

        planning = result["planning_results"]

        emergency = result["emergency_fund_plan"]

        affordability = result["combined_affordability"]

        p1, p2, p3 = st.columns(3)

        p1.metric(
            "Essential Monthly Expenses",
            f"₹{planning['average_monthly_essential_expenses']:,.2f}"
        )

        p2.metric(
            "Emergency Fund Target",
            f"₹{emergency['target_amount']:,.2f}"
        )

        p3.metric(
            "Emergency Fund Progress",
            f"{emergency['progress_percentage']:.1f}%"
        )

        st.progress(
            min(
                emergency["progress_percentage"] / 100,
                1.0
            )
        )

        
        # ========================================================
        # FINANCIAL GOAL PROGRESS AND PLANNING
        # ========================================================

        st.subheader("Financial Goal Progress and Planning")

        goal_results = planning.get("goal_contributions", [])

        if goal_results:

            goals_display = pd.DataFrame(goal_results)

            # Calculate current goal progress
            goals_display["progress_percentage"] = (
                goals_display["current_amount"]
                / goals_display["target_amount"]
            ) * 100

            goals_display["progress_percentage"] = (
                goals_display["progress_percentage"].clip(0, 100)
            )

            # Prepare readable column names
            goals_display = goals_display.rename(columns={
                "goal_name": "Financial Goal",
                "target_amount": "Target Amount (₹)",
                "current_amount": "Already Saved (₹)",
                "remaining_amount": "Remaining Amount (₹)",
                "months_to_achieve": "Timeline (Months)",
                "monthly_contribution": "Monthly Contribution (₹)",
                "priority": "Priority",
                "progress_percentage": "Progress (%)"
            })

            # Display goal details
            st.dataframe(
                goals_display,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Target Amount (₹)": st.column_config.NumberColumn(
                        format="₹%.2f"
                    ),
                    "Already Saved (₹)": st.column_config.NumberColumn(
                        format="₹%.2f"
                    ),
                    "Remaining Amount (₹)": st.column_config.NumberColumn(
                        format="₹%.2f"
                    ),
                    "Monthly Contribution (₹)": st.column_config.NumberColumn(
                        format="₹%.2f"
                    ),
                    "Progress (%)": st.column_config.ProgressColumn(
                        min_value=0,
                        max_value=100,
                        format="%.1f%%"
                    )
                }
            )

            # Combined goal contribution
            st.metric(
                "Total Monthly Goal Contribution",
                f"₹{planning['total_monthly_goal_contribution']:,.2f}"
            )

        else:

            st.info("No financial goals have been provided.")


        # ========================================================
        # COMBINED FINANCIAL AFFORDABILITY
        # ========================================================

        st.subheader("Combined Financial Affordability")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Available Monthly Cash",
            f"₹{affordability['available_monthly_cash']:,.2f}"
        )

        c2.metric(
            "Total Planned Contributions",
            f"₹{affordability['total_monthly_planned_contribution']:,.2f}"
        )

        c3.metric(
            "Remaining Monthly Cash",
            f"₹{affordability['remaining_monthly_cash']:,.2f}"
        )

        if affordability["affordability_assessment"] == "appears_affordable":
            st.success(
                "The planned goal and emergency fund contributions "
                "appear affordable based on the recorded figures."
            )
        else:
            st.warning(
                "The planned contributions exceed the available "
                "monthly cash. The plan needs to be reviewed."
            )


        # ========================================================
        # OTHER FINANCIAL PLANNING CALCULATIONS
        # ========================================================

        st.subheader("Financial Planning Calculations")

        # Avoid displaying the goal list a second time as raw nested data
        planning_summary = {
            key: value
            for key, value in planning.items()
            if key != "goal_contributions"
        }

        display_agent_output(planning_summary)


        st.subheader("Financial Planning Agent Observations")

        display_agent_output(
            result["financial_planning_agent_result"]
        )



    # ========================================================
    # TAB 3: INVESTMENT RESEARCH
    # ========================================================

    with tabs[2]:

        
        
        research = result[
            "investment_research_result"
        ]

        st.subheader("Investment Research Agent")

        st.markdown("### Research Overview")

        st.write(
            research["research_output"].research_overview
        )

        st.markdown("### Research Findings")

        display_agent_output(
            research["research_output"]
        )

        st.subheader("Supporting Investment Evidence")

        display_agent_output({
            "investment_evidence":
                research["investment_evidence"],

            "platform_evidence":
                research["platform_evidence"]
        })



    # ========================================================
    # TAB 4: RISK ASSESSMENT
    # ========================================================

    with tabs[3]:

        
        risk = result[
            "risk_assessment_result"
        ]

        st.subheader("Overall Risk Assessment")

        st.metric(
            "Risk Assessment",
            risk.overall_risk_level.replace(
                "_", " "
            ).title()
        )

        st.write(risk.risk_overview)

        st.subheader("Detailed Risk Findings")

        display_agent_output(risk)



    # ========================================================
    # TAB 5: FINANCIAL REPORT
    # ========================================================

    with tabs[4]:

        report_markdown = result[
            "financial_report_markdown"
        ]

        st.markdown(
            report_markdown
        )

        st.download_button(

            "Download Financial Report",

            data=report_markdown,

            file_name="financial_report.md",

            mime="text/markdown",

            use_container_width=True

        )


else:

    st.info(
        "Provide your transactions and financial goals, "
        "then select Run Financial Assistant."
    )
