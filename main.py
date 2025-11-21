from db.config import create_db_and_tables
from db.crud import create_user, get_user_by_id


def main():
    print("--- Database Setup ---")
    create_db_and_tables()
    print("Database and tables initialized.")

    # 1. Create a user
    print("\n--- Creating User ---")
    try:
        new_user = create_user(
            username="MIMI",
            email="MIMI.MEOW@AURA.67",
            preferred_theme="pink",
            phone_number="079676767",
        )
        print(f"Created user ID: {new_user.user_id}, Username: {new_user.username}")
    except Exception as e:
        print(f"Error during user creation: {e}")
        return

    # 2. Read the user back
    print("\n--- Reading User ---")
    retrieved_user = get_user_by_id(new_user.user_id)
    if retrieved_user:
        print(f"Retrieved user details: {retrieved_user}")


if __name__ == "__main__":
    main()
