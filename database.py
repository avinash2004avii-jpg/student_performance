import sqlite3
conn = sqlite3.connect("users.db")
cursor = conn.cursor()

# create users table
cursor.execute("""
CREATE TABLE IF NOT EXISTS users(
id INTEGER PRIMARY KEY AUTOINCREMENT,
username TEXT,
password TEXT,
role TEXT
)
""")
# insert admin user
cursor.execute(
"INSERT INTO users (username,password,role) VALUES ('admin','admin123','admin')"
)
# insert teacher user
cursor.execute(
"INSERT INTO users (username,password,role) VALUES ('teacher','teacher123','teacher')"
)
conn.commit()
conn.close()
print("Database created successfully")