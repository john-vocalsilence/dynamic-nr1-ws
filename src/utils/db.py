import atexit
from contextlib import contextmanager
from typing import Optional
from psycopg2 import pool
from settings import DB_CONFIG

db_pool: Optional[pool.ThreadedConnectionPool] = None

def init_db_pool(minconn: int = 1, maxconn: int = 5):
    """Initialize a threaded connection pool using settings.DB_CONFIG."""
    global db_pool
    if db_pool is not None:
        return

    try:
        cfg = DB_CONFIG
        db_pool = pool.ThreadedConnectionPool(
            minconn,
            maxconn,
            host=cfg["host"],
            port=cfg["port"],
            dbname=cfg["dbname"],
            user=cfg["user"],
            password=cfg["password"],
            connect_timeout=15
        )
        print("[DB] Connection pool created")
    except Exception as e:
        print(f"[DB] init_db_pool error: {e}")
        raise

def close_db_pool():
    """Close all connections in the pool."""
    global db_pool
    if db_pool:
        try:
            db_pool.closeall()
            db_pool = None
            print("[DB] Connection pool closed")
        except Exception as e:
            print(f"[DB] close_db_pool error: {e}")

atexit.register(close_db_pool)

@contextmanager
def get_db_connection():
    """
    Context manager that yields a DB connection from the pool.
    Commits on success, rollbacks on exception and returns the connection to the pool.
    """
    global db_pool
    conn = None
    try:
        if db_pool is None:
            init_db_pool()
        conn = db_pool.getconn()
        yield conn
        conn.commit()
    except Exception as e:
        if conn:
            try:
                conn.rollback()
            except Exception:
                pass
        print(f"[DB] get_db_connection error: {e}")
        raise
    finally:
        if conn and db_pool:
            db_pool.putconn(conn)

__all__ = [
    "init_db_pool", 
    "close_db_pool", 
    "get_db_connection", 
    "db_pool"
]