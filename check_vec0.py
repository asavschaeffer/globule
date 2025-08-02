import sqlite3
try:
    conn = sqlite3.connect(":memory:")
    conn.enable_load_extension(True)
    conn.load_extension("vec0")
    print("vec0 extension loaded successfully.")
except Exception as e:
    print(f"Error loading vec0 extension: {e}")