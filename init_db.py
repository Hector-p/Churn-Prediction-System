"""
Database initialization script - Creates all required tables
"""
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    print("Error: DATABASE_URL is not set in .env file")
    exit(1)

# Import models to register them
from app.database.db import Base, engine
from app.database.models import User, UsageLog, Transaction, ModelPrediction, FeatureStore

# Create all tables
print("Initializing database tables...")
Base.metadata.create_all(bind=engine)

# Verify tables were created
inspector = inspect(engine)
tables = inspector.get_table_names()

print("\nCreated/existing tables:")
for table in sorted(tables):
    columns = inspector.get_columns(table)
    print(f"  ✓ {table} ({len(columns)} columns)")

print("\n✓ Database initialization complete!")
