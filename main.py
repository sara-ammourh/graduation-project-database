from db.config import create_db_and_tables
from db.crud import create_user_auth, get_user_auth_by_id, get_all_users, update_user


def main():
    print("--- Database Setup ---")
    create_db_and_tables()
    print("Database and tables initialized.")

    # 1. Create a user_auth
    print("\n--- Creating UserAuth ---")
    try:
        new_user_auth = create_user_auth(
            password="meowwmeow",
            token="jfrflf4ol",
            user_id=1,

        )
        print(f"Created user_auth ID: {new_user_auth.user_id}, Password: {new_user_auth.password}")
    except Exception as e:
        print(f"Error during user_auth creation: {e}")
        return

    # 2. Read the user_auth back
    print("\n--- Reading UserAuth ---")
    retrieved_user_auth = get_user_auth_by_id(new_user_auth.user_id)
    if retrieved_user_auth:
        print(f"Retrieved user details: {retrieved_user_auth}")

    # # 3. Get all users
    # print("\n--- Reading All Users ---")
    # all_users = get_all_users()
    # for user in all_users:
    #     print(f"Retrieving user details: {user}")

    # # 4. Modify user
    # print("\n--- Updating User ---")
    # old_user = get_user_by_id(1)
    # if old_user:
    #     print(f"Retrieving user details: {old_user}")
    # updated_user = update_user(1, {'username': 'updated_user'})
    # if updated_user:
    #     print("User updated successfully.")
    #     print(f"Retrieving user details: {updated_user}")

if __name__ == "__main__":
    main()
