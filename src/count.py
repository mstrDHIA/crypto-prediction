import os

def get_and_increment_count(path="src/count.txt"):
    count_file = path

    # Check if the count file exists
    if not os.path.exists(count_file):
        with open(count_file, "w") as f:
            f.write("0")

    # Read the current count
    with open(count_file, "r") as f:
        count = int(f.read().strip())
        print(f"Old count: {count}")
    # Increment the count
    count += 1

    # Save the updated count
    with open(count_file, "w") as f:
        f.write(str(count))
        print(f"New count: {count}")

    return count