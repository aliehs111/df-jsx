import csv
import os

FASTAPI_CSV = "fastapi_routes.csv"  # Path to your backend-exported CSV
FRONTEND_SRC_DIR = "./client/src"   # Path to your React source folder
OUTPUT_CSV = "matched_routes.csv"   # Output file with matches

def find_endpoint_usage(endpoint_path, search_dir):
    """Search frontend files for exact or partial matches to endpoint_path."""
    matching_files = []
    dynamic_match = False

    # Strip path params like {id} for dynamic matching
    base_path = endpoint_path.split("{")[0]

    for root, _, files in os.walk(search_dir):
        for filename in files:
            if filename.endswith((".js", ".jsx", ".ts", ".tsx")):
                file_path = os.path.join(root, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        if endpoint_path in content:
                            # Exact literal match
                            rel_path = os.path.relpath(file_path, search_dir)
                            matching_files.append(rel_path)
                        elif base_path and base_path in content:
                            # Dynamic/partial match
                            rel_path = os.path.relpath(file_path, search_dir)
                            matching_files.append(rel_path)
                            dynamic_match = True
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    return matching_files, dynamic_match

def match_routes():
    with open(FASTAPI_CSV, "r", newline="") as infile, open(OUTPUT_CSV, "w", newline="") as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ["ReactFiles"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            endpoint_path = row["Path"]
            matching_files, dynamic_match = find_endpoint_usage(endpoint_path, FRONTEND_SRC_DIR)
            if matching_files:
                match_type = "MATCHED (dynamic)" if dynamic_match else "MATCHED"
                row["ReactFiles"] = f"{match_type}: " + ", ".join(matching_files)
            else:
                row["ReactFiles"] = "NOT FOUND"
            writer.writerow(row)

    print(f"✅ Matching complete. Output saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    match_routes()

