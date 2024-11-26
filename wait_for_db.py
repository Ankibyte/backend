import os
import time
import psycopg2

def wait_for_db():
    db_config = {
        'dbname': os.getenv('POSTGRES_DB', 'ankibyte_dev'),
        'user': os.getenv('POSTGRES_USER', 'ankibyte_user'),  # Changed default
        'password': os.getenv('POSTGRES_PASSWORD', 'ankibyte_password_dev'),  # Changed default
        'host': os.getenv('DB_HOST', 'db'),
        'port': os.getenv('DB_PORT', '5432')
    }

    for i in range(30):  # try for 30 seconds
        try:
            print(f"Attempting to connect to database with config: {db_config}")
            conn = psycopg2.connect(**db_config)
            conn.close()
            print("Database is ready!")
            return True
        except psycopg2.OperationalError as e:
            print(f"Waiting for database... {i+1}/30")
            print(f"Error: {str(e)}")
            time.sleep(1)
    
    raise Exception("Could not connect to database")

if __name__ == "__main__":
    wait_for_db()