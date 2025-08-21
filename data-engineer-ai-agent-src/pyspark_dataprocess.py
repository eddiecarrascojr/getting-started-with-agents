import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when

# It's recommended to set your OpenAI API key as an environment variable
# for security reasons.
# from openai import OpenAI
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# --- PySpark Setup and Core Functions ---

def initialize_spark_session():
    """
    Initializes and returns a SparkSession.
    This is the entry point to any Spark functionality.
    """
    spark = SparkSession.builder \
        .appName("AIAgentWithPySpark") \
        .master("local[*]") \
        .getOrCreate()
    print("SparkSession initialized successfully.")
    return spark

def process_csv_to_spark_db(spark, file_path, table_name):
    """
    Task 1: Reads a CSV file, processes it, and saves it as a temporary view (our "Spark database").

    Args:
        spark (SparkSession): The active SparkSession.
        file_path (str): The path to the input CSV file.
        table_name (str): The name for the temporary view to be created.

    Returns:
        str: A confirmation message.
    """
    try:
        print(f"Reading CSV from: {file_path}")
        # Read the CSV file into a DataFrame, inferring the schema and using the first row as a header.
        df = spark.read.csv(file_path, header=True, inferSchema=True)

        # Create a temporary view from the DataFrame. This view is session-scoped.
        df.createOrReplaceTempView(table_name)
        
        print(f"Successfully created temporary view: '{table_name}'")
        return f"Success: The file '{os.path.basename(file_path)}' was processed and saved as the table '{table_name}'."
    except Exception as e:
        return f"Error processing CSV: {e}"

def perform_data_quality_check(spark, table_name):
    """
    Task 2: Reads a Spark table and performs data quality checks.

    Args:
        spark (SparkSession): The active SparkSession.
        table_name (str): The name of the table to check.

    Returns:
        str: A summary of the data quality checks.
    """
    try:
        print(f"Performing data quality checks on table: '{table_name}'")
        # Check if the table exists
        if not spark.catalog.tableExists(table_name):
            return f"Error: Table '{table_name}' does not exist."

        df = spark.table(table_name)
        total_rows = df.count()
        
        # 1. Check for null values in each column
        null_counts = df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).collect()[0].asDict()
        null_report = [f"  - Column '{k}': {v} null values" for k, v in null_counts.items() if v > 0]
        
        # 2. Check for duplicate rows
        duplicate_count = total_rows - df.distinct().count()

        # Build the report string
        report = f"Data Quality Report for table '{table_name}':\n"
        report += f"- Total Rows: {total_rows}\n"
        report += f"- Duplicate Rows Found: {duplicate_count}\n"
        if null_report:
            report += "- Null Value Counts:\n" + "\n".join(null_report)
        else:
            report += "- No null values found in any column.\n"
            
        print(report)
        return report
    except Exception as e:
        return f"Error during data quality check: {e}"

def create_joined_view(spark, table1_name, table2_name, join_column, new_view_name):
    """
    Task 3: Joins two tables and creates a new view from the result.

    Args:
        spark (SparkSession): The active SparkSession.
        table1_name (str): The name of the first table.
        table2_name (str): The name of the second table.
        join_column (str): The name of the column to join on.
        new_view_name (str): The name for the new joined view.

    Returns:
        str: A confirmation message.
    """
    try:
        print(f"Joining '{table1_name}' and '{table2_name}' on column '{join_column}'")
        # Check if both tables exist
        if not spark.catalog.tableExists(table1_name):
            return f"Error: Table '{table1_name}' does not exist."
        if not spark.catalog.tableExists(table2_name):
            return f"Error: Table '{table2_name}' does not exist."

        table1_df = spark.table(table1_name)
        table2_df = spark.table(table2_name)

        # Perform an inner join
        joined_df = table1_df.join(table2_df, on=join_column, how="inner")

        # Create a new temporary view from the joined DataFrame
        joined_df.createOrReplaceTempView(new_view_name)
        
        print(f"Successfully created joined view: '{new_view_name}'")
        return f"Success: Created new view '{new_view_name}' by joining '{table1_name}' and '{table2_name}'."
    except Exception as e:
        return f"Error creating joined view: {e}"

# --- Main Execution Block ---

def main():
    """
    Main function to demonstrate the agent's tasks.
    """
    # Initialize Spark
    spark = initialize_spark_session()

    # --- Create Dummy Data for Demonstration ---
    # In a real scenario, these files would already exist.
    # We create them here to make the script self-contained.
    
    # Dataset 1: Employees
    employees_data = """employee_id,employee_name,department_id
101,Alice,1
102,Bob,2
103,Charlie,1
104,David,3
105,Eve,2
"""
    employees_csv_path = "employees.csv"
    with open(employees_csv_path, "w") as f:
        f.write(employees_data)

    # Dataset 2: Departments
    departments_data = """department_id,department_name
1,Engineering
2,Marketing
3,Finance
"""
    departments_csv_path = "departments.csv"
    with open(departments_csv_path, "w") as f:
        f.write(departments_data)
        
    print("\nCreated dummy CSV files: 'employees.csv' and 'departments.csv'")

    # --- Simulate AI Agent Executing Tasks ---
    
    print("\n--- Task 1: Processing CSVs into Spark DB ---")
    # Agent receives a request to process the employee data
    task1_result_employees = process_csv_to_spark_db(spark, employees_csv_path, "employees_table")
    print(task1_result_employees)
    
    # Agent receives a request to process the department data
    task1_result_departments = process_csv_to_spark_db(spark, departments_csv_path, "departments_table")
    print(task1_result_departments)

    print("\n--- Task 2: Performing Data Quality Checks ---")
    # Agent is asked to run a quality check on the new employees table
    task2_result = perform_data_quality_check(spark, "employees_table")
    print(task2_result)

    print("\n--- Task 3: Creating a Joined View ---")
    # Agent is asked to create a combined view of employees and their department names
    task3_result = create_joined_view(
        spark,
        table1_name="employees_table",
        table2_name="departments_table",
        join_column="department_id",
        new_view_name="employee_details_view"
    )
    print(task3_result)
    
    print("\n--- Verifying the final joined view ---")
    # Let's see the result of the final view
    spark.table("employee_details_view").show()

    # --- Clean up dummy files ---
    os.remove(employees_csv_path)
    os.remove(departments_csv_path)
    print("\nCleaned up dummy CSV files.")

    # Stop the SparkSession
    spark.stop()
    print("SparkSession stopped.")

if __name__ == "__main__":
    # To run this, you need to have pyspark installed:
    # pip install pyspark openai
    main()

# --- OpenAI Agent Integration (Conceptual) ---
#
# The functions above (`process_csv_to_spark_db`, `perform_data_quality_check`, `create_joined_view`)
# are designed to be used as "tools" by an OpenAI Assistant.
#
# Here's how you would set that up:
#
# 1. Define the tools in the format OpenAI expects:
#
# tools_list = [
#     {
#         "type": "function",
#         "function": {
#             "name": "process_csv_to_spark_db",
#             "description": "Reads a CSV file and loads it into a Spark database table.",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "file_path": {"type": "string", "description": "The local path to the CSV file."},
#                     "table_name": {"type": "string", "description": "The desired name for the Spark table."}
#                 },
#                 "required": ["file_path", "table_name"]
#             }
#         }
#     },
#     # ... (define other functions similarly)
# ]
#
# 2. Create the Assistant with the tools:
#
# assistant = client.beta.assistants.create(
#     instructions="You are a data assistant. You use PySpark tools to manage data.",
#     model="gpt-4-turbo-preview",
#     tools=tools_list
# )
#
# 3. When a user makes a request (e.g., "Process the file 'employees.csv' and name the table 'employees'"),
#    the model will recognize the need to call your function and respond with a `tool_calls` object.
#
# 4. Your application code would then execute the actual Python function (e.g., `process_csv_to_spark_db(...)`)
#    and submit the result back to the Assistant, which will then formulate the final response to the user.
