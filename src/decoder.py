# This file contains routines from Lisbon Machine Learning summer school.
# The code is freely distributed under a MIT license. https://github.com/LxMLS/lxmls-toolkit/

import torch
import numpy as np
import pdb

def log_add_exp(a, b):
    max_ab = torch.max(a, b)
    # max_ab[~isfinite(max_ab)] = 0
    return torch.log(torch.add(torch.exp(a - max_ab), torch.exp(b - max_ab))) + max_ab

def log_sum_exp(tensor, dim=1):
    # Compute log sum exp in a numerically stable way for the forward algorithm

    xmax, _ = torch.max(tensor, dim = dim, keepdim = True)
    xmax_, _ = torch.max(tensor, dim = dim)
    return xmax_ + torch.log(torch.sum(torch.exp(tensor - xmax), dim = dim))

def parse_proj(scores, gold=None):
    '''
    Parse using Eisner's algorithm.
    '''
    nr, nc = np.shape(scores)
    if nr != nc:
        raise ValueError("scores must be a squared matrix with nw+1 rows")

    N = nr - 1  # Number of words (excluding root).
    L, R = 0, 1

    # Initialize CKY table.
    complete = np.zeros([N + 1, N + 1, 2])  # s, t, direction (right=1).
    incomplete = np.zeros([N + 1, N + 1, 2])  # s, t, direction (right=1).
    complete_backtrack = -np.ones([N + 1, N + 1, 2], dtype=int)  # s, t, direction (right=1).
    incomplete_backtrack = -np.ones([N + 1, N + 1, 2], dtype=int)  # s, t, direction (right=1).

    incomplete[0, :, L] -= np.inf
    scores[:,0] = -np.inf
    for i in range(N+1):
        scores[i, i] = -np.inf

    # Loop from smaller items to larger items.
    for k in xrange(1, N + 1):
        for s in xrange(N - k + 1):
            t = s + k

            # First, create incomplete items.
            # left tree
            incomplete_vals0 = complete[s, s:t, R] + complete[(s + 1):(t + 1), t, L] + scores[t, s] + (0.0 if (gold is not None and gold[s] == t) else 1.0)
            incomplete[s, t, L] = np.max(incomplete_vals0)
            incomplete_backtrack[s, t, L] = s + np.argmax(incomplete_vals0)
            # right tree
            incomplete_vals1 = complete[s, s:t, R] + complete[(s + 1):(t + 1), t, L] + scores[s, t] + (0.0 if (gold is not None and gold[t] == s) else 1.0)
            incomplete[s, t, R] = np.max(incomplete_vals1)
            incomplete_backtrack[s, t, R] = s + np.argmax(incomplete_vals1)

            # Second, create complete items.
            # left tree
            complete_vals0 = complete[s, s:t, L] + incomplete[s:t, t, L]
            complete[s, t, L] = np.max(complete_vals0)
            complete_backtrack[s, t, L] = s + np.argmax(complete_vals0)
            # right tree
            complete_vals1 = incomplete[s, (s + 1):(t + 1), R] + complete[(s + 1):(t + 1), t, R]
            complete[s, t, R] = np.max(complete_vals1)
            complete_backtrack[s, t, R] = s + 1 + np.argmax(complete_vals1)

    value = complete[0][N][R]
    heads = [-1 for _ in range(N + 1)]  # -np.ones(N+1, dtype=int)
    backtrack_eisner(incomplete_backtrack, complete_backtrack, 0, N, 1, 1, heads)

    value_proj = 0.0
    for m in xrange(1, N + 1):
        h = heads[m]
        value_proj += scores[h, m]

    return heads

def parse_nonproj(scores, verbose = False):
        """
        Parse using Chu-Liu-Edmonds algorithm.
        """
        nr, nc = np.shape(scores)
        if nr != nc:
            raise ValueError("scores must be a squared matrix with N+1 rows")
            return []

        N = nr - 1

        curr_nodes = np.ones(N+1, int)
        reps = []
        old_I = -np.ones((N+1, N+1), int)
        old_O = -np.ones((N+1, N+1), int)
        for i in range(0, N+1):
            reps.append({i: 0})
            for j in range(0, N+1):
                old_I[i, j] = i
                old_O[i, j] = j
                if i == j or j == 0:
                    continue

        if verbose:
            print("Starting C-L-E...\n")

        scores_copy = scores.copy()
        final_edges = chu_liu_edmonds(scores_copy, curr_nodes, old_I, old_O, {}, reps)
        heads = np.zeros(N+1, int)
        heads[0] = -1
        for key in list(final_edges.keys()):
            ch = key
            pr = final_edges[key]
            heads[ch] = pr

        return heads

def chu_liu_edmonds(scores, curr_nodes, old_I, old_O, final_edges, reps, verbose = False):
        """
        Chu-Liu-Edmonds algorithm
        """

        # need to construct for each node list of nodes they represent (here only!)
        nw = np.size(curr_nodes) - 1

        # create best graph
        par = -np.ones(nw+1, int)
        for m in range(1, nw+1):
            # only interested in current nodes
            if 0 == curr_nodes[m]:
                continue
            max_score = scores[0, m]
            par[m] = 0
            for h in range(nw+1):
                if m == h:
                    continue
                if 0 == curr_nodes[h]:
                    continue
                if scores[h, m] > max_score:
                    max_score = scores[h, m]
                    par[m] = h

        if verbose:
            print("After init\n")
            for m in range(0, nw+1):
                if 0 < curr_nodes[m]:
                    print("{0}|{1} ".format(par[m], m))
            print("\n")

        # find a cycle
        cycles = []
        added = np.zeros(nw+1, int)
        for m in range(0, nw+1):
            if np.size(cycles) > 0:
                break
            if added[m] or 0 == curr_nodes[m]:
                continue
            added[m] = 1
            cycle = {m: 0}
            l = m
            while True:
                if par[l] == -1:
                    added[l] = 1
                    break
                if par[l] in cycle:
                    cycle = {}
                    lorg = par[l]
                    cycle[lorg] = par[lorg]
                    added[lorg] = 1
                    l1 = par[lorg]
                    while l1 != lorg:
                        cycle[l1] = par[l1]
                        added[l1] = True
                        l1 = par[l1]
                    cycles.append(cycle)
                    break
                cycle[l] = 0
                l = par[l]
                if added[l] and (l not in cycle):
                    break
                added[l] = 1

        # get all edges and return them
        if np.size(cycles) == 0:
            for m in range(0, nw+1):
                if 0 == curr_nodes[m]:
                    continue
                if par[m] != -1:
                    pr = old_I[par[m], m]
                    ch = old_O[par[m], m]
                    final_edges[ch] = pr
                else:
                    final_edges[0] = -1
            return final_edges

        max_cyc = 0
        wh_cyc = 0
        for cycle in cycles:
            if np.size(list(cycle.keys())) > max_cyc:
                max_cyc = np.size(list(cycle.keys()))
                wh_cyc = cycle

        cycle = wh_cyc
        cyc_nodes = sorted(list(cycle.keys()))
        rep = cyc_nodes[0]

        if verbose:
            print("Found Cycle\n")
            for node in cyc_nodes:
                print("{0} ".format(node))
            print("\n")

        cyc_weight = 0.0
        for node in cyc_nodes:
            cyc_weight += scores[par[node], node]

        for i in range(0, nw+1):
            if 0 == curr_nodes[i] or (i in cycle):
                continue

            max1 = -np.inf
            wh1 = -1
            max2 = -np.inf
            wh2 = -1

            for j1 in cyc_nodes:
                if scores[j1, i] > max1:
                    max1 = scores[j1, i]
                    wh1 = j1

                # cycle weight + new edge - removal of old
                scr = cyc_weight + scores[i, j1] - scores[par[j1], j1]
                if scr > max2:
                    max2 = scr
                    wh2 = j1

            scores[rep, i] = max1
            old_I[rep, i] = old_I[wh1, i]
            old_O[rep, i] = old_O[wh1, i]
            scores[i, rep] = max2
            old_O[i, rep] = old_O[i, wh2]
            old_I[i, rep] = old_I[i, wh2]

        rep_cons = []
        for i in range(0, np.size(cyc_nodes)):
            rep_con = {}
            keys = sorted(reps[int(cyc_nodes[i])].keys())
            if verbose:
                print("{0}: ".format(cyc_nodes[i]))
            for key in keys:
                rep_con[key] = 0
                if verbose:
                    print("{0} ".format(key))
            rep_cons.append(rep_con)
            if verbose:
                print("\n")

        # don't consider not representative nodes
        # these nodes have been folded
        for node in cyc_nodes[1:]:
            curr_nodes[node] = 0
            for key in reps[int(node)]:
                reps[int(rep)][key] = 0

        chu_liu_edmonds(scores, curr_nodes, old_I, old_O, final_edges, reps)

        # check each node in cycle, if one of its representatives
        # is a key in the final_edges, it is the one.
        if verbose:
            print(final_edges)
        wh = -1
        found = False
        for i in range(0, np.size(rep_cons)):
            if found:
                break
            for key in rep_cons[i]:
                if found:
                    break
                if key in final_edges:
                    wh = cyc_nodes[i]
                    found = True
        l = par[wh]
        while l != wh:
            ch = old_O[par[l]][l]
            pr = old_I[par[l]][l]
            final_edges[ch] = pr
            l = par[l]

        return final_edges

def viterbi(scores):
    '''
    Parse using Eisner's algorithm.
    '''
    nr, nc = np.shape(scores)
    if nr != nc:
        raise ValueError("scores must be a squared matrix with nw+1 rows")

    N = nr - 1  # Number of words (excluding root).
    L, R = 0, 1

    # Initialize CKY table.
    complete = np.zeros([N + 1, N + 1, 2])  # s, t, direction (right=1).
    incomplete = np.zeros([N + 1, N + 1, 2])  # s, t, direction (right=1).
    complete_backtrack = -np.ones([N + 1, N + 1, 2], dtype=int)  # s, t, direction (right=1).
    incomplete_backtrack = -np.ones([N + 1, N + 1, 2], dtype=int)  # s, t, direction (right=1).

    incomplete[0, :, L] -= np.inf

    # Loop from smaller items to larger items.
    for k in xrange(1, N + 1):
        for s in xrange(N - k + 1):
            t = s + k

            # First, create incomplete items.
            # left tree
            incomplete_vals0 = complete[s, s:t, R] + complete[(s + 1):(t + 1), t, L] + scores[t, s]
            incomplete[s, t, L] = np.max(incomplete_vals0)
            incomplete_backtrack[s, t, L] = s + np.argmax(incomplete_vals0)
            # right tree
            incomplete_vals1 = complete[s, s:t, R] + complete[(s + 1):(t + 1), t, L] + scores[s, t]
            incomplete[s, t, R] = np.max(incomplete_vals1)
            incomplete_backtrack[s, t, R] = s + np.argmax(incomplete_vals1)

            # Second, create complete items.
            # left tree
            complete_vals0 = complete[s, s:t, L] + incomplete[s:t, t, L]
            complete[s, t, L] = np.max(complete_vals0)
            complete_backtrack[s, t, L] = s + np.argmax(complete_vals0)
            # right tree
            complete_vals1 = incomplete[s, (s + 1):(t + 1), R] + complete[(s + 1):(t + 1), t, R]
            complete[s, t, R] = np.max(complete_vals1)
            complete_backtrack[s, t, R] = s + 1 + np.argmax(complete_vals1)

    value = complete[0, N, R]
    heads = [-1 for _ in range(N + 1)]  # -np.ones(N+1, dtype=int)
    backtrack_eisner(incomplete_backtrack, complete_backtrack, 0, N, 1, 1, heads)

    value_proj = 0.0
    for m in xrange(1, N + 1):
        h = heads[m]
        value_proj += scores[h, m]

    return heads

def inside(scores):
    # scores: a torch matrix tensor
    '''
    inside algorithm
    '''
    # pdb.set_trace()
    nr, nc = scores.shape
    if nr != nc:
        raise ValueError("scores must be a squared matrix with nw+1 rows")

    N = nr - 1  # Number of words (excluding root).
    L, R = 0, 1

    # Initialize CKY tables.
    if not torch.cuda.is_available():
        complete = torch.zeros([N + 1, N + 1, 2]).double()  # len * len * directions
        incomplete = torch.zeros([N + 1, N + 1, 2]).double()  # len * len * directions
    else:
        complete = torch.zeros([N + 1, N + 1, 2]).double().cuda()  # len * len * directions
        incomplete = torch.zeros([N + 1, N + 1, 2]).double().cuda()  # len * len * directions

    incomplete.fill_(-np.inf)
    complete.fill_(-np.inf)

    for i in range(N+1):
        complete[i, i, L] = 0.0
        complete[i, i, R] = 0.0

    # Loop from smaller spans to larger spans.
    # pdb.set_trace()
    for k in xrange(1, N + 1): # k is the distance between i and j
        for s in xrange(N - k + 1): # s is the start
            t = s + k # t is the end

            # First, create incomplete spans.
            # left tree
            incomplete_vals0 = complete[s, s:t, R] + complete[(s + 1):(t + 1), t, L] + scores[t, s]
            incomplete[s, t, L] = torch.logsumexp(incomplete_vals0, 0)

            # right tree
            incomplete_vals1 = complete[s, s:t, R] + complete[(s + 1):(t + 1), t, L] + scores[s, t]
            incomplete[s, t, R] = torch.logsumexp(incomplete_vals1, 0)


            # Second, create complete spans.
            # left tree
            complete_vals0 = complete[s, s:t, L] + incomplete[s:t, t, L]
            complete[s, t, L] = torch.logsumexp(complete_vals0, 0)

            # right tree
            complete_vals1 = incomplete[s, (s + 1):(t + 1), R] + complete[(s + 1):(t + 1), t, R]
            complete[s, t, R] = torch.logsumexp(complete_vals1, 0)


    partition_value = complete[0, N, R]

    # pdb.set_trace()
    return partition_value

def backtrack_eisner(incomplete_backtrack, complete_backtrack, s, t, direction, complete, heads):
    '''
    Backtracking step in Eisner's algorithm.
    - incomplete_backtrack is a (NW+1)-by-(NW+1) numpy array indexed by a start position,
    an end position, and a direction flag (0 means left, 1 means right). This array contains
    the arg-maxes of each step in the Eisner algorithm when building *incomplete* spans.
    - complete_backtrack is a (NW+1)-by-(NW+1) numpy array indexed by a start position,
    an end position, and a direction flag (0 means left, 1 means right). This array contains
    the arg-maxes of each step in the Eisner algorithm when building *complete* spans.
    - s is the current start of the span
    - t is the current end of the span
    - direction is 0 (left attachment) or 1 (right attachment)
    - complete is 1 if the current span is complete, and 0 otherwise
    - heads is a (NW+1)-sized numpy array of integers which is a placeholder for storing the
    head of each word.
    '''
    L, R = 0, 1
    if s == t:
        return
    if complete:
        r = complete_backtrack[s][t][direction]
        if direction == L:
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 0, 1, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r, t, 0, 0, heads)
            return
        else:
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 0, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r, t, 1, 1, heads)
            return
    else:
        r = incomplete_backtrack[s][t][direction]
        if direction == L:
            heads[s] = t
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 1, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r + 1, t, 0, 1, heads)
            return
        else:
            heads[t] = s
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 1, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r + 1, t, 0, 1, heads)
            return
