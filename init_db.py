from database import engine, Base
import models # Crucial: This imports the classes so SQLAlchemy knows they exist

print("Connecting to PostgreSQL to build the schema...")

# This command inspects the database. 
# If a table does not exist, it runs the CREATE TABLE SQL command.
# If the table already exists, it safely ignores it.
Base.metadata.create_all(bind=engine)

print("Relational tables created successfully!")