import subprocess
import sys

def run_git_commands(commit_message):
    try:
        # Run git add .
        subprocess.run(['git', 'add', '.'], check=True)
        print("Changes staged.")

        # Run git commit with provided message
        subprocess.run(['git', 'commit', '-m', commit_message], check=True)
        print(f"Changes committed with message: '{commit_message}'")

        # Run git push
        subprocess.run(['git', 'push', 'origin', 'main'], check=True)
        print("Changes pushed to remote repository.")

    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Prompt the user for a commit message
    commit_message = input("Enter your commit message: ")

    # Run the git commands
    run_git_commands(commit_message)
