import psycopg2
from psycopg2 import sql

# Database credentials
DB_NAME = "rag_3"  # Replace with your database name
DB_USER = "postgres"
DB_PASSWORD = "example"
DB_HOST = "localhost"
DB_PORT = "5432"

def drop_database(db_name):
    try:
        # Connect to the default 'postgres' database
        conn = psycopg2.connect(
            dbname="postgres",
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        conn.autocommit = True  # Required to execute DROP DATABASE statement
        cursor = conn.cursor()

        # Terminate all connections to the target database
        terminate_query = sql.SQL("""
            SELECT
                pg_terminate_backend(pid)
            FROM
                pg_stat_activity
            WHERE
                datname = %s
                AND pid <> pg_backend_pid();
        """)
        cursor.execute(terminate_query, (db_name,))
        print(f"Terminated all connections to the database '{db_name}'.")

        # Drop the target database
        drop_db_query = sql.SQL("DROP DATABASE IF EXISTS {}").format(
            sql.Identifier(db_name)
        )
        cursor.execute(drop_db_query)
        print(f"Database '{db_name}' dropped successfully.")

        # Close the connection
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error dropping database '{db_name}': {e}")

if __name__ == "__main__":
    drop_database(DB_NAME)