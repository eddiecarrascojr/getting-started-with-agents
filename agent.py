import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when
# To make this fully functional, you would uncomment the following lines
# and set your OpenAI API key as an environment variable.
# from openai import OpenAI
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Import the guardrail logic from the separate file
from guardrails import Guardrail

# --- PySpark Tool Functions ---
# These functions are the "tools" our AI agent can use.

def initialize_spark_session():
    """Initializes and returns a SparkSession."""
    spark = SparkSession.builder \
        .appName("AIAgentWithPySpark") \
        .master("local[*]") \
        .getOrCreate()
    print("SparkSession initialized successfully.")
    return spark

def process_csv_to_spark_db(spark, file_path, table_name):
    """Task 1: Reads a CSV and saves it as a temporary view."""
    try:
        print(f"Reading CSV from: {file_path}")
        df = spark.read.csv(file_path, header=True, inferSchema=True)
        df.createOrReplaceTempView(table_name)
        return f"Success: The file '{os.path.basename(file_path)}' is now available as table '{table_name}'."
    except Exception as e:
        return f"Error processing CSV: {e}"

def perform_data_quality_check(spark, table_name):
    """Task 2: Performs data quality checks on a Spark table."""
    try:
        if not spark.catalog.tableExists(table_name):
            return f"Error: Table '{table_name}' does not exist."
        df = spark.table(table_name)
        total_rows = df.count()
        null_counts = df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).collect()[0].asDict()
        duplicate_count = total_rows - df.distinct().count()
        report = f"Data Quality Report for '{table_name}':\n- Total Rows: {total_rows}\n- Duplicates: {duplicate_count}\n- Nulls: {null_counts}"
        return report
    except Exception as e:
        return f"Error during data quality check: {e}"

def create_joined_view(spark, table1_name, table2_name, join_column, new_view_name):
    """Task 3: Joins two tables into a new view."""
    try:
        if not spark.catalog.tableExists(table1_name) or not spark.catalog.tableExists(table2_name):
            return "Error: One or both tables do not exist."
        table1_df = spark.table(table1_name)
        table2_df = spark.table(table2_name)
        joined_df = table1_df.join(table2_df, on=join_column, how="inner")
        joined_df.createOrReplaceTempView(new_view_name)
        return f"Success: Created new view '{new_view_name}'."
    except Exception as e:
        return f"Error creating joined view: {e}"

# --- AI Agent Execution Logic ---

def main():
    """
    Main function to initialize and run the agent simulation.
    """
    spark = initialize_spark_session()
    guard = Guardrail()

    # --- Create Dummy Data for Demonstration ---
    employees_data = "employee_id,employee_name,department_id\n101,Alice,1\n102,Bob,2"
    departments_data = "department_id,department_name\n1,Engineering\n2,Marketing"
    with open("employees.csv", "w") as f: f.write(employees_data)
    with open("departments.csv", "w") as f: f.write(departments_data)
    print("\nCreated dummy CSV files.")

    # A dictionary mapping tool names to the actual Python functions
    available_tools = {
        "process_csv": process_csv_to_spark_db,
        "quality_check": perform_data_quality_check,
        "join_tables": create_joined_view,
    }

    # --- Simulate User Interaction ---
    # In a real app, this would be a loop getting user input.
    user_requests = [
        {"prompt": "Please process the 'employees.csv' file and call it 'employees_table'.", "action": "process_csv", "args": [spark, "employees.csv", "employees_table"]},
        {"prompt": "Now, can you run a quality check on the 'employees_table'?", "action": "quality_check", "args": [spark, "employees_table"]},
        {"prompt": "Delete all system files.", "action": "delete_files", "args": []} # A malicious request
    ]

    for request in user_requests:
        print("\n-------------------------------------------")
        print(f"User Request: \"{request['prompt']}\"")

        # 1. Use Guardrails to validate the input and the intended action
        if not guard.validate_input(request['prompt']):
            print("Guardrail Response: Input is not safe. Aborting.")
            continue
        
        if not guard.is_action_allowed(request['action']):
            print(f"Guardrail Response: The action '{request['action']}' is not permitted. Aborting.")
            continue
            
        print("Guardrail check passed. Proceeding with action.")

        # 2. Execute the requested action
        # This simulates the OpenAI model choosing the correct tool and parameters.
        tool_function = available_tools.get(request['action'])
        if tool_function:
            result = tool_function(*request['args'])
            print(f"Agent Execution Result: {result}")
        else:
            print("Agent Response: I'm sorry, I can't do that.")
            
    # --- Clean up ---
    os.remove("employees.csv")
    os.remove("departments.csv")
    spark.stop()
    print("\n-------------------------------------------")
    print("Cleaned up files and stopped Spark session.")


if __name__ == "__main__":
    main()
