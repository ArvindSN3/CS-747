import argparse
import pulp
import numpy as np

def parse_mdp(file_path):
    mdp = {}
    with open(file_path, "r") as file:
        lines = file.read().splitlines()
        for line in lines:
            parts = line.split()

            if parts[0] == "numStates":
                mdp["S"] = int(parts[1])
            elif parts[0] == "numActions":
                mdp["A"] = int(parts[1])
                mdp["T"] = np.zeros((mdp["S"], mdp["A"], mdp["S"]))
                mdp["R"] = np.zeros((mdp["S"], mdp["A"], mdp["S"]))
            elif parts[0] == "end":
                mdp["end"] = list(map(int, parts[1:]))
            elif parts[0] == "transition":
                s1 = int(parts[1])
                act = int(parts[2])
                s2 = int(parts[3])
                r = float(parts[4])
                p = float(parts[5])
                mdp["T"][s1, act, s2] = p
                mdp["R"][s1, act, s2] = r
            elif parts[0] == "mdptype":
                mdp["mdptype"] = parts[1]
            elif parts[0] == "discount":
                mdp["Y"] = float(parts[1])

    return mdp

def parse_policy(file_path):
    pol = []
    with open(file_path, "r") as file:
        lines = file.read().splitlines()
        for line in lines:
            pol.append(int(line))
        
        policy = np.array(pol)

        return policy

def value_eval(mdp, policy):
    R = mdp["R"][np.arange(mdp["S"]), policy]
    T = mdp["T"][np.arange(mdp["S"]), policy]
    
    value = np.squeeze(np.linalg.inv(np.eye(mdp["S"]) - mdp["Y"] * T)
                       @ np.sum(T * R, axis=-1, keepdims=True))
    return value


def val_iter(mdp):
    value = np.zeros(mdp["S"])
    policy = np.zeros(mdp["S"], dtype=int)

    while True:
        max_diff = 0
        new_value = np.zeros(mdp["S"])
        for s in range(mdp["S"]):
            old_v = value[s]
            q_values = np.sum(mdp["T"][s, :, :] * (mdp["R"][s, :, :] + mdp["Y"] * value), axis=-1)
            new_value[s] = np.max(q_values)
            policy[s] = np.argmax(q_values)
            max_diff = max(max_diff, np.abs(new_value[s] - old_v))

        value = new_value

        if max_diff < 1e-9:
            break

    return value, policy

def howard_PI(mdp):
    policy = np.random.randint(low=0, high=mdp["A"], size=mdp["S"])

    while True:
        max_diff = 0
        value = np.zeros(mdp["S"])

        while True:
            max_diff = 0
            new_V = np.zeros(mdp["S"])
            for s in range(mdp["S"]):
                old_v = value[s]
                a = policy[s]
                q_values = np.sum(mdp["T"][s, a, :] * (mdp["R"][s, a, :] + mdp["Y"] * value))
                new_V[s] = q_values
                max_diff = max(max_diff, np.abs(new_V[s] - old_v))

            value = new_V

            if max_diff < 1e-9:
                break

        policy_new = np.argmax(np.sum(mdp["T"] * (mdp["R"] + mdp["Y"] * value), axis=-1), axis=-1)

        if np.array_equal(policy_new, policy):
            break

        policy = policy_new

    return value, policy

def LP(mdp):
    prob = pulp.LpProblem("MDP_LP", pulp.LpMinimize)

    value = np.array(list(pulp.LpVariable.dicts("Value", [i for i in range(mdp["S"])], lowBound=0).values()))
    
    prob += pulp.lpSum(value)

    for s in range(mdp["S"]):
        for a in range(mdp["A"]):
            transition = np.sum(mdp["T"][s, a, :] * (mdp["R"][s, a, :] + mdp["Y"] * value))
            prob += value[s] >= transition

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    value = np.array([pulp.value(value[i]) for i in range(mdp["S"])])
    policy = np.argmax(np.sum(mdp["T"] * (mdp["R"] + mdp["Y"] * value.reshape(1, 1, -1)), axis=-1), axis=-1)

    return value, policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mdp", required = True)
    parser.add_argument("--algorithm", default = "lp")
    parser.add_argument("--policy", default = "None")

    args = parser.parse_args()

    mdp_path = args.mdp
    algorithm = args.algorithm
    policy_path = args.policy

    parsed_mdp = parse_mdp(mdp_path)

    if policy_path == "None":
        if algorithm == "vi":
            opt_val, opt_pol = val_iter(parsed_mdp)
        if algorithm == "hpi":
            opt_val, opt_pol = howard_PI(parsed_mdp)
        if algorithm == "lp":
            opt_val, opt_pol = LP(parsed_mdp)
        
        for i in range(len(opt_val)):
            print('{:.6f}'.format(round(opt_val[i], 6)) + "\t" + str(int(opt_pol[i])))
    
    else:
        parsed_policy = parse_policy(policy_path)
        pol_val = value_eval(parsed_mdp, parsed_policy)

        for i in range(len(pol_val)):
            print('{:.6f}'.format(round(pol_val[i], 6)) + "\t" + str(int(parsed_policy[i])))