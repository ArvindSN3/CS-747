import argparse
import sys

def decode_planner_output(output_file, opponent_file):
    # Read the opponent file to determine the order of positions
    with open(opponent_file, "r") as opp_file:
        positions = [int(line.strip()) for line in opp_file]

    # Read the output from the planner
    with open(output_file, "r") as planner_output:
        lines = planner_output.read().splitlines()

    policy_data = []
    for i, line in enumerate(lines):
        value, action = line.split()
        position = positions[i]
        policy_data.append((float(value), int(action), position))

    # Sort policy data based on the original position
    policy_data.sort(key=lambda x: x[2])

    # Write the policy data to stdout (console)
    for data in policy_data:
        sys.stdout.write(f"{data[0]:.6f} {data[1]}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--planner-output", required=True)
    parser.add_argument("--opponent-file", required=True)

    args = parser.parse_args()

    decode_planner_output(args.planner_output, args.opponent_file)