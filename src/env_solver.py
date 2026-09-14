import numpy as np

import scipy
from scipy.sparse.linalg import spsolve, factorized


class SparseMDPSolver:
    def __init__(
        self,
        trans_probs,
        rewards,
        initial_state_probs,
        observations,
        discount_factor=0.999,
        max_eval_iters=100,
        tolerance=1e-8,
        improve_tolerance=1e-10,
        max_iters=10000,
        feature_names=None,
        action_names=None,
    ):
        assert trans_probs.ndim == 3
        assert initial_state_probs.ndim == 1
        assert observations.ndim == 2
        assert trans_probs.shape == rewards.shape
        assert trans_probs.shape[0] == trans_probs.shape[2]
        assert initial_state_probs.shape[0] == trans_probs.shape[0]
        assert observations.shape[0] == trans_probs.shape[0]

        if isinstance(trans_probs, np.ndarray):
            self.sparse_trans_probs = self.__dense_to_sparse_trans_probs(trans_probs)
        else:
            assert isinstance(trans_probs, list)
            assert all(
                isinstance(maxtrix, scipy.sparse.csr_matrix) for maxtrix in trans_probs
            )
            self.sparse_trans_probs = trans_probs

        self.rewards = rewards
        self.initial_state_probs = initial_state_probs
        self.observations = observations
        self.discount_factor = discount_factor
        self.max_eval_iters = max_eval_iters
        self.tolerance = tolerance
        self.improve_tolerance = improve_tolerance
        self.max_iters = max_iters

        self.n_actions_ = len(self.sparse_trans_probs)
        self.n_states_ = self.sparse_trans_probs[0].shape[0]

        if rewards.shape == (self.n_states_, self.n_actions_):
            assert isinstance(rewards, np.ndarray), type(rewards)
            self.expected_rewards_ = rewards
        else:
            self.expected_rewards_ = self.__compute_expected_rewards()

        if feature_names:
            self.feature_names_ = feature_names
        else:
            self.feature_names_ = [f"x{i}" for i in range(observations.shape[1])]
        
        if action_names:
            self.action_names_ = action_names
        else:
            self.action_names_ = [f"action{i}" for i in range(self.n_actions_)]

    def get_observations(self):
        return self.observations

    def __dense_to_sparse_trans_probs(self, trans_probs):
        n_actions = trans_probs.shape[1]
        return [scipy.sparse.csr_matrix(trans_probs[:, a, :]) for a in range(n_actions)]

    def __compute_expected_rewards(self):
        n_states, n_actions = self.rewards.shape[:2]
        expected_rewards = np.zeros((n_states, n_actions))
        for a in range(n_actions):
            expected_rewards[:, a] = (
                (self.sparse_trans_probs[a].multiply(self.rewards[:, a, :]))
                .sum(axis=-1)
                .ravel()
            )

        return expected_rewards

    def __compute_expected_rewards_fixed_actions(self, fixed_state_actions):
        expected_rewards = np.copy(self.expected_rewards_)

        if fixed_state_actions:
            for state in fixed_state_actions:
                allowed_actions = fixed_state_actions[state]
                for other_action in range(self.n_actions_):
                    if other_action not in allowed_actions:
                        expected_rewards[state, other_action] = -1e10

        return expected_rewards

    def solve(self, fixed_state_actions=None, initial_values=None, initial_policy=None):
        values, policy = self.__solve_policy_iteration_sparse_optimized(
            fixed_state_actions, initial_values, initial_policy, return_policy=True
        )
        objective = np.dot(self.initial_state_probs, values)
        return values, policy, objective
    
    def allowed_actions(self, state_id):
        # In our current MDPs every action is available in every state
        return set(range(self.n_actions_))
    
    def feasible_actions(self, state_id):
        # In our current MDPs every action is available in every state
        return self.allowed_actions(state_id)

    def __solve_policy_iteration_sparse_optimized(
        self,
        fixed_state_actions=None,
        initial_values=None,
        initial_policy=None,
        return_policy=False,
    ):
        """
        Optimized policy-iteration for sparse MDPs.

        Key idea:
        - build P_pi once per policy (fast grouping + vstack)
        - solve (I - gamma * P_pi) v = r_pi either with spsolve (direct) if small enough,
            otherwise with bicgstab (iterative) using previous v as initial guess.
        Returns: (values, policy)
        """

        expected_rewards = self.__compute_expected_rewards_fixed_actions(
            fixed_state_actions
        )

        # initial policy
        if initial_policy is None:
            policy = np.zeros(self.n_states_, dtype=np.int32)
        else:
            policy = np.copy(initial_policy)

        if initial_values is None:
            values = np.zeros(self.n_states_)
        else:
            values = np.copy(initial_values)

        for it in range(self.max_iters):
            # Build policy-specific transition matrix once
            P_pi = self.__build_policy_transition_from_sparse_list(policy)

            # RHS: expected rewards under policy
            r_pi = expected_rewards[np.arange(self.n_states_), policy]

            # Solve (I - gamma * P_pi) v = r_pi
            A = (
                scipy.sparse.identity(self.n_states_, format="csr")
                - self.discount_factor * P_pi
            )

            A_csc = A.tocsc()
            new_values = spsolve(A_csc, r_pi)

            values = np.asarray(new_values, dtype=float)

            # Policy improvement: compute q(s,a) = r(s,a) + gamma * (P_a @ values)
            # Compute Pv for all actions, but only for actions that exist or needed.
            # If many actions but only a few states choose them, consider computing only required ones.
            Pv = np.empty((self.n_actions_, self.n_states_), dtype=float)
            for a in range(self.n_actions_):
                Pv[a] = self.sparse_trans_probs[a].dot(values)

            # Prevent oscillations by only changing to strictly improving actions
            q = expected_rewards.T + self.discount_factor * Pv  # (A,S)
            best_actions = np.argmax(q, axis=0)
            best_q = q[best_actions, np.arange(self.n_states_)]
            current_q = q[policy, np.arange(self.n_states_)]
            new_policy = policy.copy()
            better = best_q > current_q + self.improve_tolerance
            new_policy[better] = best_actions[better]

            if np.array_equal(new_policy, policy):
                break

            policy = new_policy

        # Final (more accurate) evaluation of final policy:
        P_pi = self.__build_policy_transition_from_sparse_list(policy)
        r_pi = expected_rewards[np.arange(self.n_states_), policy]
        A = (
            scipy.sparse.identity(self.n_states_, format="csr")
            - self.discount_factor * P_pi
        )
        values = spsolve(A.tocsc(), r_pi)

        if return_policy:
            return values, policy

        return values

    def __build_policy_transition_from_sparse_list(self, policy):
        """
        Build P_pi (csr) whose i-th row is row i from sparse_trans_probs[policy[i]].
        Implementation:
        - group states by action,
        - vstack each group's rows from the corresponding P_a,
        - reorder the stacked rows to the natural state order.
        This does ONE vstack per action present in the policy.
        """

        blocks = []
        row_order_parts = []
        for a in range(self.n_actions_):
            idx = np.nonzero(policy == a)[0]
            if idx.size:
                # sparse_trans_probs[a][idx] -> block with rows in the order of idx
                blocks.append(
                    self.sparse_trans_probs[a][idx]
                )  # slicing a CSR by row gives CSR
                row_order_parts.append(idx)

        if not blocks:
            # no states? return empty square csr
            return scipy.sparse.csr_matrix((self.n_states_, self.n_states_))

        P_block = scipy.sparse.vstack(blocks, format="csr")
        row_order = np.concatenate(
            row_order_parts
        )  # maps P_block row -> original state index
        # perm such that row_order[perm] == np.arange(n_states)
        perm = np.argsort(row_order)
        P_pi = P_block[perm]  # reorder rows so row i corresponds to state i
        assert P_pi.shape == (self.n_states_, self.n_states_)
        return P_pi

    def values_to_q_values(
        self,
        values: np.ndarray,
        fixed_state_actions=None,
    ):
        expected_rewards = self.__compute_expected_rewards_fixed_actions(
            fixed_state_actions
        )

        Pv = np.empty((self.n_states_, self.n_actions_), dtype=float)
        for a in range(self.n_actions_):
            Pv[:, a] = self.sparse_trans_probs[a].dot(values)
        q_values = expected_rewards + self.discount_factor * Pv
        return q_values

    def compute_discounted_state_dist(
        self,
        policy: np.ndarray,
    ):
        """
        Solves: d = (1-gamma) * mu0 + gamma * d P^π
        """
        # TODO: possibly cache P_pi from calls to other functions. Maybe take it as optional param?
        P_pi = self.__build_policy_transition_from_sparse_list(policy)

        # Build once in CSC for solves
        A = (
            scipy.sparse.identity(self.n_states_, format="csc")
            - self.discount_factor * P_pi.T.tocsc()
        )

        solver = factorized(A)
        # return (1 - self.discount_factor) * solver(self.initial_state_probs)
        return solver(self.initial_state_probs)

    def evaluate_policy(
        self,
        policy: np.ndarray,
    ):
        P_pi = self.__build_policy_transition_from_sparse_list(policy)
        r_pi = self.expected_rewards_[np.arange(self.n_states_), policy]
        n_states = len(policy)

        A = scipy.sparse.identity(n_states, format="csr") - self.discount_factor * P_pi
        values = spsolve(A.tocsc(), r_pi)

        return np.dot(values, self.initial_state_probs)


def solve_value_iteration(
    trans_probs: np.ndarray,
    rewards: np.ndarray,
    discount_factor=0.999,
    tolerance=1e-8,
    max_iters=10000,
    initial_values=None,
    fixed_state_actions=None,
):
    n_states = trans_probs.shape[0]
    n_actions = trans_probs.shape[1]

    if initial_values is None:
        values = np.zeros(n_states, dtype=trans_probs.dtype)
    else:
        values = np.copy(initial_values)

    # Pre-compute expected rewards once (unchanged)
    expected_rewards = np.einsum("sak,sak->sa", trans_probs, rewards)

    if fixed_state_actions:
        for state in fixed_state_actions:
            allowed_actions = fixed_state_actions[state]
            for other_action in range(n_actions):
                if other_action not in allowed_actions:
                    expected_rewards[state, other_action] = -1e10

    # Pre-allocate arrays to avoid repeated allocation
    q_values = np.empty((n_states, trans_probs.shape[1]))

    for _ in range(max_iters):
        # Use optimized einsum with pre-allocated output
        np.einsum("sak,k->sa", trans_probs, values, out=q_values)
        q_values *= discount_factor
        q_values += expected_rewards

        # Compute new values and convergence check in one pass
        new_values = q_values.max(axis=1)
        max_error = np.abs(values - new_values).max()

        if max_error < tolerance:
            return new_values

        values = new_values

    return values


def values_to_policy(
    values: np.ndarray,
    trans_probs: np.ndarray,
    rewards: np.ndarray,
    discount_factor=0.999,
    fixed_state_actions=None,
):
    expected_rewards = np.einsum("sak,sak->sa", trans_probs, rewards)
    n_actions = trans_probs.shape[1]
    if fixed_state_actions:
        for state in fixed_state_actions:
            allowed_actions = fixed_state_actions[state]
            for other_action in range(n_actions):
                if other_action not in allowed_actions:
                    expected_rewards[state, other_action] = -1e10
    q_values = expected_rewards + np.einsum(
        "sak,sak->sa", trans_probs, discount_factor * values[np.newaxis, np.newaxis, :]
    )

    policy = np.argmax(q_values, axis=1)

    return policy


def values_to_q_values(
    values: np.ndarray,
    trans_probs: np.ndarray,
    rewards: np.ndarray,
    discount_factor=0.999,
    fixed_state_actions=None,
):
    expected_rewards = np.einsum("sak,sak->sa", trans_probs, rewards)
    n_actions = trans_probs.shape[1]
    if fixed_state_actions:
        for state in fixed_state_actions:
            allowed_actions = fixed_state_actions[state]
            for other_action in range(n_actions):
                if other_action not in allowed_actions:
                    expected_rewards[state, other_action] = -1e10
    q_values = expected_rewards + np.einsum(
        "sak,sak->sa", trans_probs, discount_factor * values[np.newaxis, np.newaxis, :]
    )
    return q_values


def find_reachable_states(
    trans_probs: np.ndarray, initial_state_probs=np.ndarray, fixed_state_actions=None
):
    n_states, n_actions = trans_probs.shape[:2]
    reachable = np.zeros(n_states, dtype=bool)
    stack = list(np.nonzero(initial_state_probs)[0])
    while stack:
        state = stack.pop()
        reachable[state] = True

        for action in range(n_actions):
            if (
                state in fixed_state_actions
                and action not in fixed_state_actions[state]
            ):
                continue

            for next_state in np.nonzero(trans_probs[state, action])[0]:
                if not reachable[next_state]:
                    stack.append(next_state)
    return reachable


def find_reachable_states_policy(
    trans_probs: np.ndarray, initial_state_probs: np.ndarray, policy: np.ndarray
):
    n_states = trans_probs.shape[0]
    reachable = np.zeros(n_states, dtype=bool)
    stack = list(np.nonzero(initial_state_probs)[0])
    while stack:
        state = stack.pop()
        reachable[state] = True

        for next_state in np.nonzero(trans_probs[state, policy[state]])[0]:
            if not reachable[next_state]:
                stack.append(next_state)
    return reachable


def compute_discounted_state_dist(
    trans_probs: np.ndarray,
    pi: np.ndarray,
    initial_state_probs: np.ndarray,
    discount_factor=0.999,
    tolerance=1e-8,
    max_iters=10000,
):
    """
    Solves: d = (1-gamma) * mu0 + gamma * d P^π
    """
    n_states = trans_probs.shape[0]
    P_pi = trans_probs[np.arange(n_states), pi, :]

    d = initial_state_probs.copy()

    for _ in range(max_iters):
        new_d = (1 - discount_factor) * initial_state_probs + discount_factor * d @ P_pi

        if np.max(np.abs(new_d - d)) < tolerance:
            return new_d

        d = new_d

    return d


def policy_to_reward_process(policy, trans_probs, rewards):
    n_states = trans_probs.shape[0]

    P_pi = np.zeros((n_states, n_states))
    r_pi = np.zeros(n_states)

    for s in range(n_states):
        a = policy[s]
        P_sa = trans_probs[s, a, :]
        R_sa = rewards[s, a, :]

        P_pi[s, :] = P_sa
        r_pi[s] = np.sum(P_sa * R_sa)

    return P_pi, r_pi


def evaluate_policy(
    policy: np.ndarray,
    trans_probs: np.ndarray,
    rewards: np.ndarray,
    discount_factor=0.999,
    tol=1e-8,
    max_iters=10000,
):
    P_pi, r_pi = policy_to_reward_process(policy, trans_probs, rewards)
    n_states = len(policy)

    values = np.zeros(n_states)

    for _ in range(max_iters):
        new_values = r_pi + discount_factor * P_pi @ values
        if np.max(np.abs(new_values - values)) < tol:
            break
        values = new_values

    return values


def evaluate_random_policy(
    trans_probs: np.ndarray,
    rewards: np.ndarray,
    discount_factor=0.999,
    tol=1e-8,
    max_iters=10000,
):
    P_pi = trans_probs.mean(axis=1)
    r_pi = np.sum(P_pi * rewards.mean(axis=1), axis=1)

    n_states = trans_probs.shape[0]
    values = np.zeros(n_states)

    for _ in range(max_iters):
        new_values = r_pi + discount_factor * P_pi @ values
        if np.max(np.abs(new_values - values)) < tol:
            break
        values = new_values

    return values
