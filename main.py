from db.config import create_db_and_tables
from db.crud import create_user, get_user_by_id, get_all_users, update_user


def main():
    print("--- Database Setup ---")
    create_db_and_tables()
    print("Database and tables initialized.")

    # # 1. Create a user
    # print("\n--- Creating User ---")
    # try:
    #     new_user = create_user(
    #         username="MIMI",
    #         email="MIMI.MEOW@AURA.67",
    #         preferred_theme="pink",
    #         phone_number="079676767",
    #     )
    #     print(f"Created user ID: {new_user.user_id}, Username: {new_user.username}")
    # except Exception as e:
    #     print(f"Error during user creation: {e}")
    #     return

    # # 2. Read the user back
    # print("\n--- Reading User ---")
    # retrieved_user = get_user_by_id(new_user.user_id)
    # if retrieved_user:
    #     print(f"Retrieved user details: {retrieved_user}")

    # 3. Get all users
    print("\n--- Reading All Users ---")
    all_users = get_all_users()
    for user in all_users:
        print(f"Retrieving user details: {user}")

    # 4. Modify user
    print("\n--- Updating User ---")
    old_user = get_user_by_id(1)
    if old_user:
        print(f"Retrieving user details: {old_user}")
    updated_user = update_user(1, {'username': 'updated_user'})
    if updated_user:
        print("User updated successfully.")
        print(f"Retrieving user details: {updated_user}")

if __name__ == "__main__":
    main()
