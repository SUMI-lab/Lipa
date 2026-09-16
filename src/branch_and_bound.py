import time

import numpy as np

from sklearn.tree import DecisionTreeClassifier, export_text

from pystreed import STreeDInstanceCostSensitiveClassifier, STreeDClassifier

from .env_solver import (
    SparseMDPSolver,
)

from .storm_solver import StormSolver

from typing import Union

import random

random.seed(1)


def try_solve_exact_actions(
    observations, state_dist, optimal_actions, n_states, max_depth, time_limit
):
    constrained_states = []
    for state in range(n_states):
        if state_dist[state] > 0.0:
            constrained_states.append(state)
    # obs = observations[constrained_states][50:-25]
    # actions = optimal_actions[constrained_states][50:-25]
    obs = observations[constrained_states]
    actions = optimal_actions[constrained_states]

    heur_tree = DecisionTreeClassifier(max_depth=max_depth, random_state=1)
    heur_tree.fit(obs, actions)
    heuristic_score = heur_tree.score(obs, actions)
    if heuristic_score == 1.0:
        return heur_tree

    exact_tree = STreeDClassifier(
        max_depth=max_depth,
        time_limit=time_limit,
        n_categories=10000000,
        n_thresholds=10000000,
        cost_complexity=0.0,
    )
    exact_tree.fit(obs, actions)

    if not exact_tree.fit_result.is_optimal():
        print("Exact solve attempt did not prove optimality within the time limit")

    if heuristic_score > exact_tree.score(obs, actions):
        return heur_tree

    return exact_tree


def solve_tree_branch_and_bound(
    solver: Union[SparseMDPSolver, StormSolver],
    max_depth,
    time_limit=None,
    optimality_tol=1e-4,
    track_progress=False,
    skip_branch_heuristic=False,
    skip_node_selection_heuristic=False,
    skip_all_tree_heuristics=False,
    skip_perfect_tree_heuristic=False,
):
    if track_progress:
        scores = []

    start_time = time.time()

    if time_limit:
        max_time = start_time + time_limit
    else:
        max_time = float("inf")

    if solver.n_states_ >= 1000000:
        do_q_value_heuristic = False
        skip_perfect_tree_heuristic = True
        print("Skipping Q-value heuristic and perfect tree due to huge state space")
    else:
        do_q_value_heuristic = True

    initial_values, initial_actions, relaxed_objective = solver.solve()

    bound = relaxed_objective
    if track_progress:
        scores.append((time.time() - start_time, -float("inf"), relaxed_objective))

    if time.time() > max_time:
        return None, relaxed_objective, None

    print("Relaxed optimum:", relaxed_objective)

    # TODO: if random actions are available, ignore states with 1 action + terminal states

    observations = solver.get_observations()
    states_allowed_actions = [
        solver.allowed_actions(state_i) for state_i in range(solver.n_states_)
    ]
    states_feasible_actions = states_allowed_actions

    if skip_all_tree_heuristics:
        heuristic_initial_tree = DecisionTreeClassifier(
            max_depth=max_depth, random_state=1
        ).fit(
            observations,
            [next(iter(states_allowed_actions[state])) for state in range(solver.n_states_)],
        )
    else:
        heuristic_initial_tree = DecisionTreeClassifier(
            max_depth=max_depth, random_state=1
        ).fit(
            observations,
            initial_actions,
        )
    realizable = all(
        pred in allowed
        for pred, allowed in zip(
            heuristic_initial_tree.predict(observations),
            states_allowed_actions,
            strict=True,
        )
    )
    if realizable:
        heur_predicted_actions = heuristic_initial_tree.predict(observations)
        heur_objective = solver.evaluate_policy(heur_predicted_actions)
        current_best = heur_objective
        best_tree = heuristic_initial_tree
        if track_progress:
            scores.append(
                (time.time() - start_time, current_best, relaxed_objective)
            )
        print("initial tree objective (heur)", current_best)

        if time.time() > max_time:
            return current_best, relaxed_objective, best_tree
    else:
        print("couldn't find heuristic feasible tree, trying optimal...")
        exact_initial_tree = STreeDInstanceCostSensitiveClassifier(
            max_depth=max_depth,
            time_limit=10000000,
            n_categories=10000000,
            n_thresholds=10000000,
        )
        costs = np.ones((solver.n_states_, solver.n_actions_))
        for state in range(solver.n_states_):
            for action in states_allowed_actions[state]:
                costs[state, action] = 0.0
        exact_initial_tree.fit(observations, costs)
        realizable = np.isclose(
            costs[
                np.arange(costs.shape[0]), exact_initial_tree.predict(observations)
            ].sum(),
            0.0,
        )
        if not realizable:
            print("MDP has no feasible tree!")
            return -np.inf, relaxed_objective, None

        exact_predicted_actions = exact_initial_tree.predict(observations)
        exact_objective = solver.evaluate_policy(exact_predicted_actions)
        current_best = exact_objective
        best_tree = exact_initial_tree
        if track_progress:
            scores.append(
                (time.time() - start_time, current_best, relaxed_objective)
            )
        print("initial tree objective", current_best)

        if time.time() > max_time:
            return current_best, relaxed_objective, best_tree

    if do_q_value_heuristic:
        all_actions = []
        all_states = []
        for state in range(solver.n_states_):
            for action in states_allowed_actions[state]:
                all_states.append(state)
                all_actions.append(action)
        all_actions = np.array(all_actions)
        all_states = np.array(all_states)

    # Tuples of:
    # - branch priority
    # - relaxed objective
    # - state values
    # - optimal actions
    # - allowed actions for each state (dict of sets, allow all if not defined)
    # - whether to skip heuristics
    queue = [(-np.inf, relaxed_objective, initial_values, initial_actions, {}, False)]

    last_log_time = start_time
    nodes_processed = 0
    current_time = time.time()
    while queue:
        current_time = time.time()
        if current_time > max_time:
            break

        bound = max(max(queue, key=lambda x: x[1])[1], relaxed_objective)
        if current_best == -float("inf") or current_best == 0.0:
            if bound == 0.0:
                gap = 0.0
            else:
                gap = float("inf")
        else:
            gap = abs(bound - current_best) / abs(current_best)
        if gap <= optimality_tol:
            break

        if track_progress:
            scores.append((current_time - start_time, current_best, bound))

        if current_time - last_log_time > 10:
            last_log_time = current_time
            print(
                f"{last_log_time - start_time:.1f}s | Processed {nodes_processed} nodes, queue size {len(queue)}, current_best {current_best:.4f}, bound {bound:.4f}, gap {gap:.2%}"
            )

        queue = [item for item in queue if item[1] > current_best]
        if len(queue) == 0:
            break

        if skip_node_selection_heuristic:
            random.shuffle(queue)
        else:
            queue = sorted(queue, key=lambda x: x[0])
        (
            _,
            relaxed_objective,
            optimal_values,
            optimal_actions,
            fixed_state_actions_sub,
            skip_heuristics,
        ) = queue.pop()
        nodes_processed += 1

        if current_best is not None and relaxed_objective < current_best - 1e-5:
            continue

        q_values = solver.values_to_q_values(
            optimal_values, fixed_state_actions=fixed_state_actions_sub
        )
        q_values -= q_values.min()

        if do_q_value_heuristic:
            q_values_heur = np.array(
                [q_values[s, a] for s, a in zip(all_states, all_actions)]
            )

        state_dist = solver.compute_discounted_state_dist(optimal_actions)

        if len(fixed_state_actions_sub) == 0 and not skip_perfect_tree_heuristic:
            # In the first iteration try to fit an optimal tree
            time_lim = float("inf") if time_limit is None else time_limit
            exact_attempt_tree = try_solve_exact_actions(
                observations,
                state_dist,
                optimal_actions,
                solver.n_states_,
                max_depth,
                min(0.01 * time_lim, 5),
            )
            exact_attempt_actions = exact_attempt_tree.predict(observations)
            exact_attempt_objective = solver.evaluate_policy(exact_attempt_actions)
            if exact_attempt_objective > current_best:
                if exact_attempt_objective == relaxed_objective:
                    if track_progress:
                        scores.append(
                            (
                                time.time() - start_time,
                                exact_attempt_objective,
                                relaxed_objective,
                            )
                        )
                    return (
                        exact_attempt_objective,
                        relaxed_objective,
                        exact_attempt_tree,
                    )
                current_best = exact_attempt_objective
                best_tree = exact_attempt_tree
            print("exact attempt tree objective", exact_attempt_objective)

            if time.time() > max_time:
                break

        if not skip_heuristics and not skip_all_tree_heuristics:
            best_iter_tree = None
            best_iter_tree_score = -np.inf
            for name, fit_tree in (
                (
                    "CART I",
                    lambda: DecisionTreeClassifier(
                        max_depth=max_depth, random_state=1
                    ).fit(
                        observations,
                        optimal_actions,
                        sample_weight=state_dist
                        * (optimal_values - optimal_values.min()),
                    ),
                ),
                (
                    "CART Q",
                    lambda: DecisionTreeClassifier(
                        max_depth=max_depth, random_state=1
                    ).fit(
                        observations[all_states],
                        all_actions,
                        sample_weight=q_values_heur,
                    ),
                ),
            ):
                if not do_q_value_heuristic and name == "CART Q":
                    continue

                heuristic_tree = fit_tree()

                # evaluate the tree here on the non-transformed MDP to get a new heuristic score
                predicted_actions = heuristic_tree.predict(observations)
                heuristic_objective = solver.evaluate_policy(predicted_actions)

                if current_best is None or heuristic_objective > current_best:
                    print("new best:", heuristic_objective, "using", name)
                    current_best = heuristic_objective
                    best_tree = heuristic_tree

                if heuristic_objective >= best_iter_tree_score:
                    best_iter_tree_score = heuristic_objective
                    best_iter_tree = heuristic_tree

                if time.time() > max_time:
                    break
            heuristic_tree = best_iter_tree
            heuristic_objective = best_iter_tree_score
            predicted_actions = heuristic_tree.predict(observations)
        elif len(fixed_state_actions_sub) < 2:
            heuristic_tree = DecisionTreeClassifier(
                max_depth=max_depth, random_state=1
            ).fit(
                observations,
                optimal_actions,
                sample_weight=state_dist * optimal_values,
            )
            predicted_actions = heuristic_tree.predict(observations)
            heuristic_objective = solver.evaluate_policy(predicted_actions)
        else:
            heuristic_objective = -np.inf

        if abs(heuristic_objective - relaxed_objective) < 1e-6:
            continue

        if current_best is None or heuristic_objective > current_best:
            print("new best:", heuristic_objective)
            current_best = heuristic_objective
            best_tree = heuristic_tree

        if time.time() > max_time:
            break

        state_branch_heuristic = []
        for state_id in range(solver.n_states_):
            if state_id in fixed_state_actions_sub:
                filtered_q_values = q_values[
                    state_id, list(fixed_state_actions_sub[state_id])
                ]
            else:
                filtered_q_values = q_values[state_id]

            if len(filtered_q_values) > 1:
                if skip_branch_heuristic:
                    state_branch_heuristic.append(0.0)
                else:
                    state_branch_heuristic.append(
                        state_dist[state_id]
                        * (filtered_q_values.max() - filtered_q_values.min())
                    )
            else:
                state_branch_heuristic.append(-np.inf)

        diff_i = None
        if len(fixed_state_actions_sub) >= 2:
            constrained_states = []
            costs = []
            for state in range(solver.n_states_):
                if state in fixed_state_actions_sub:
                    constrained_states.append(state)
                    costs_state = np.ones(solver.n_actions_)
                    for action in fixed_state_actions_sub[state]:
                        costs_state[action] = 0.0
                    costs.append(costs_state)
                else:
                    if len(states_allowed_actions[state]) == solver.n_actions_:
                        continue
                    else:
                        constrained_states.append(state)
                        costs_state = np.ones(solver.n_actions_)
                        for action in states_allowed_actions[state]:
                            costs_state[action] = 0.0
                        costs.append(costs_state)
            costs = np.array(costs)
            obs = observations[constrained_states]

            exact_tree = STreeDInstanceCostSensitiveClassifier(
                max_depth=max_depth,
                time_limit=10000000,
                n_categories=10000000,
                n_thresholds=10000000,
            )
            exact_tree.fit(obs, costs)
            realizable = np.isclose(
                costs[np.arange(costs.shape[0]), exact_tree.predict(obs)].sum(), 0.0
            )

            if not realizable:
                continue

            exact_predicted_actions = exact_tree.predict(observations)
            exact_objective = solver.evaluate_policy(exact_predicted_actions)

            if current_best is None or exact_objective > current_best:
                print("new best:", exact_objective)
                current_best = exact_objective
                best_tree = exact_tree

            if time.time() > max_time:
                break

            branch_state_i = None
            best_branch_heuristic = -np.inf
            ignore_states = set(
                [
                    state
                    for state in fixed_state_actions_sub
                    if len(fixed_state_actions_sub[state]) == 1
                ]
            ) | set(
                state
                for state in range(solver.n_states_)
                if len(states_feasible_actions[state]) == 1
            )
            for diff_i, (optimal_action, tree_action) in enumerate(
                zip(optimal_actions, exact_predicted_actions)
            ):
                if diff_i in ignore_states:
                    continue
                if optimal_action != tree_action:
                    heur = state_branch_heuristic[diff_i]
                    if heur > best_branch_heuristic:
                        best_branch_heuristic = heur
                        branch_state_i = diff_i
            if branch_state_i is None:
                # tree and optimal are the same
                continue
        else:
            branch_state_i = None
            best_branch_heuristic = -np.inf
            ignore_states = set(
                [
                    state
                    for state in fixed_state_actions_sub
                    if len(fixed_state_actions_sub[state]) == 1
                ]
            ) | set(
                state
                for state in range(solver.n_states_)
                if len(states_feasible_actions[state]) == 1
            )
            for diff_i, (optimal_action, tree_action) in enumerate(
                zip(optimal_actions, predicted_actions)
            ):
                if diff_i in ignore_states:
                    continue
                if optimal_action != tree_action:
                    heur = state_branch_heuristic[diff_i]
                    if heur > best_branch_heuristic:
                        best_branch_heuristic = heur
                        branch_state_i = diff_i
            if branch_state_i is None:
                continue

        if branch_state_i is None:
            raise RuntimeError("No branch_state_i")

        branch_action = int(optimal_actions[branch_state_i])
        if (
            branch_state_i in fixed_state_actions_sub
            and branch_action not in fixed_state_actions_sub[branch_state_i]
        ):
            branch_action = list(fixed_state_actions_sub[branch_state_i])[0]
        elif branch_action not in states_feasible_actions[branch_state_i]:
            branch_action = list(states_feasible_actions[branch_state_i])[0]

        new_fixed_state_actions_opt = fixed_state_actions_sub.copy()
        new_fixed_state_actions_opt[branch_state_i] = set([branch_action])
        queue.append(
            (
                0.0001 * len(new_fixed_state_actions_opt) + relaxed_objective,
                relaxed_objective,
                optimal_values,
                optimal_actions,
                new_fixed_state_actions_opt,
                True,
            )
        )

        new_fixed_state_actions_notopt = fixed_state_actions_sub.copy()
        if branch_state_i in new_fixed_state_actions_notopt:
            new_fixed_state_actions_notopt[branch_state_i] = (
                new_fixed_state_actions_notopt[branch_state_i] - set([branch_action])
            )
        else:
            new_fixed_state_actions_notopt[branch_state_i] = states_feasible_actions[
                branch_state_i
            ] - set([branch_action])
        (
            new_optimal_values_notopt,
            new_optimal_actions_notopt,
            new_relaxed_objective_notopt,
        ) = solver.solve(
            fixed_state_actions=new_fixed_state_actions_notopt,
            initial_values=optimal_values,
            initial_policy=optimal_actions,
        )
        queue.append(
            (
                0.0001 * len(new_fixed_state_actions_notopt)
                + new_relaxed_objective_notopt,
                new_relaxed_objective_notopt,
                new_optimal_values_notopt,
                new_optimal_actions_notopt,
                new_fixed_state_actions_notopt,
                False,
            )
        )

    if len(queue) == 0:
        bound = current_best
    else:
        bound = max(queue, key=lambda x: x[1])[1]
    
    if track_progress:
        scores.append((time.time() - start_time, current_best, bound))
        return current_best, bound, best_tree, scores
    
    return current_best, bound, best_tree


class OptimalMDPTree:
    def __init__(
        self,
        max_depth=None,
        time_limit=None,
        track_progress=False,
        skip_branch_heuristic=False,
        skip_node_selection_heuristic=False,
        skip_all_tree_heuristics=False,
        skip_perfect_tree_heuristic=False,
    ):
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.track_progress = track_progress
        self.skip_branch_heuristic = skip_branch_heuristic
        self.skip_node_selection_heuristic = skip_node_selection_heuristic
        self.skip_all_tree_heuristics = skip_all_tree_heuristics
        self.skip_perfect_tree_heuristic = skip_perfect_tree_heuristic

    def solve(self, mdp_solver):
        if self.track_progress:
            self.score_, self.bound_, self.tree_, self.history_ = (
                solve_tree_branch_and_bound(
                    mdp_solver,
                    self.max_depth,
                    self.time_limit,
                    skip_branch_heuristic=self.skip_branch_heuristic,
                    skip_node_selection_heuristic=self.skip_node_selection_heuristic,
                    skip_all_tree_heuristics=self.skip_all_tree_heuristics,
                    skip_perfect_tree_heuristic=self.skip_perfect_tree_heuristic,
                    track_progress=True,
                )
            )
        else:
            self.score_, self.bound_, self.tree_ = solve_tree_branch_and_bound(
                mdp_solver,
                self.max_depth,
                self.time_limit,
                skip_branch_heuristic=self.skip_branch_heuristic,
                skip_node_selection_heuristic=self.skip_node_selection_heuristic,
                skip_all_tree_heuristics=self.skip_all_tree_heuristics,
                skip_perfect_tree_heuristic=self.skip_perfect_tree_heuristic,
                track_progress=False,
            )
        print("Best score:", self.score_)

        if self.score_ == 0.0:
            if self.bound_ == 0.0:
                gap = 0.0
            else:
                gap = float("inf")
        else:
            gap = abs(self.bound_ - self.score_) / abs(self.score_)

        self.optimal_ = gap <= 1e-4
        self.gap_ = gap
        self.feature_names_ = mdp_solver.feature_names_
        self.action_names_ = mdp_solver.action_names_

    def to_string(self):
        if not hasattr(self, "tree_"):
            raise RuntimeError("No fitted tree")

        if isinstance(self.tree_, DecisionTreeClassifier):
            sklearn_tree = self.tree_.tree_

            def tree_to_string_rec(node_id, depth=0):
                if (
                    sklearn_tree.children_left[node_id]
                    == sklearn_tree.children_right[node_id]
                ):
                    tree_action_id = np.argmax(sklearn_tree.value[node_id])
                    action_id = self.tree_.classes_[tree_action_id]
                    return f"{depth * '  '}{self.action_names_[action_id]}"

                left_string = tree_to_string_rec(
                    sklearn_tree.children_left[node_id], depth + 1
                )
                right_string = tree_to_string_rec(
                    sklearn_tree.children_right[node_id], depth + 1
                )
                indentation = depth * "  "
                feature_name = self.feature_names_[sklearn_tree.feature[node_id]]
                threshold = sklearn_tree.threshold[node_id]
                return f"{indentation}if {feature_name} <= {threshold}\n{left_string}\n{indentation}else:\n{right_string}"

            return tree_to_string_rec(0)
        else:
            binary_features = self.tree_.binarizer_.binary_columns
            # NOTE: we swap orders of children later since binary features actually represent "feature > 0.5"
            predicate_strs = [
                f"{self.feature_names_[col]} <= 0.5"
                for col in binary_features
            ]

            if self.tree_.binarizer_.continuous_binarizer is not None:
                for i, thresholds in enumerate(
                    self.tree_.binarizer_.continuous_binarizer.thresholds_
                ):
                    feature_i = len(self.tree_.binarizer_.binary_columns) + i
                    for threshold in thresholds:
                        predicate_strs.append(
                            f"{self.feature_names_[feature_i]} <= {threshold}"
                        )

            def tree_to_string_rec(node, depth=0):
                if node.is_leaf_node():
                    return f"{depth * '  '}{self.action_names_[node.label]}"

                left_string = tree_to_string_rec(node.left_child, depth + 1)
                right_string = tree_to_string_rec(node.right_child, depth + 1)
                indentation = depth * "  "

                if node.feature in binary_features:
                    return f"{indentation}if {predicate_strs[node.feature]}\n{left_string}\n{indentation}else:\n{right_string}"

                # NOTE: we swap orders of children since binary features actually represent "feature > 0.5"
                return f"{indentation}if {predicate_strs[node.feature]}\n{right_string}\n{indentation}else:\n{left_string}"

            streed_tree = self.tree_.get_tree()
            return tree_to_string_rec(streed_tree)
