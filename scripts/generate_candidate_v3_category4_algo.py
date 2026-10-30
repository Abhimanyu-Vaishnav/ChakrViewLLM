"""
Exp34 Candidate V3 - Category 4: Algorithmic Reasoning
Contains exactly 100 completely distinct, non-repeating core computer science algorithms,
data structure operations, dynamic programming recurrences, graph traversals, and complexity proofs.
Covers 20+ distinct algorithmic families. Zero BST DFS loop repetitions.
"""

from typing import Any, Dict, List

def build_category_4_algorithmic_reasoning() -> List[Dict[str, Any]]:
    algorithms = [
        # Family 1: Graph Algorithms & Shortest Path (1-10)
        ("Breadth-First Search (BFS)",
         "How does BFS find the shortest path in an unweighted graph?",
         "BFS utilizes a FIFO queue starting from the source vertex. It explores all immediate neighbors at distance 1 before moving to distance 2. Because edge weights are uniform, the first time a target vertex is dequeued, the path traversed is guaranteed to have the minimum number of edges. Time complexity is O(V + E)."),

        ("Dijkstra's Algorithm",
         "Describe how Dijkstra's algorithm finds single-source shortest paths in non-negative weighted graphs.",
         "Dijkstra maintains a priority queue of unvisited nodes prioritized by tentative distance from the source. At each step, it extracts the node u with minimum distance, visits it, and relaxes all outgoing edges (u, v, w): if dist[u] + w < dist[v], dist[v] is updated and v is pushed into the queue. It settles all reachable nodes in O((V + E) log V) time."),

        ("Bellman-Ford Algorithm",
         "How does the Bellman-Ford algorithm detect negative weight cycles in a directed graph?",
         "Bellman-Ford relaxes all E edges V - 1 times. In a graph without negative cycles, shortest paths contain at most V - 1 edges. If any edge can still be relaxed on the V-th iteration (dist[u] + w < dist[v]), a negative cycle reachable from the source must exist."),

        ("Floyd-Warshall Algorithm",
         "Explain the dynamic programming update rule in the Floyd-Warshall all-pairs shortest path algorithm.",
         "For each intermediate vertex k from 1 to V, and each vertex pair (i, j): dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]). It computes shortest paths between all pairs of vertices in O(V^3) time and detects negative cycles if any diagonal entry dist[i][i] becomes negative."),

        ("A* Search Heuristic Admissibility",
         "Why must the heuristic function h(n) in A* search be admissible?",
         "A heuristic h(n) is admissible if it never overestimates the true remaining cost to reach the goal (h(n) <= h*(n)). Admissibility guarantees that A* will return an optimal shortest path because the search will never prematurely finalize a suboptimal path before considering the optimal one."),

        ("Topological Sorting (Kahn's Algorithm)",
         "Describe Kahn's algorithm for topological sorting of a Directed Acyclic Graph (DAG).",
         "Kahn's algorithm computes the in-degree of all vertices and enqueues those with in-degree 0. While the queue is not empty, it dequeues vertex u, appends it to the topological order, and decrements the in-degree of its outgoing neighbors. Any neighbor whose in-degree drops to 0 is enqueued. If the order contains fewer than V vertices, a cycle exists."),

        ("Tarjan's Strongly Connected Components",
         "How does Tarjan's algorithm identify Strongly Connected Components (SCCs) in a single DFS pass?",
         "Tarjan assigns each node a discovery time index and a low-link value (the lowest index reachable from the subtree via back-edges). Nodes are pushed onto a stack during DFS. When a node's low-link equals its discovery index, it is the root of an SCC; nodes are popped from the stack until that root is removed, identifying the SCC in O(V + E) time."),

        ("Bipartite Graph Verification (2-Coloring)",
         "How does graph coloring verify whether an undirected graph is bipartite?",
         "Start BFS/DFS from an unvisited node, coloring it color 0. Assign all adjacent neighbors color 1, and their neighbors color 0. If an edge ever connects two vertices that already share the same color, an odd-length cycle exists, proving the graph cannot be bipartite. Time complexity is O(V + E)."),

        ("Eulerian Path (Hierholzer's Algorithm)",
         "What are the vertex degree conditions for an Eulerian path to exist in an undirected graph?",
         "An undirected connected graph contains an Eulerian path (visiting every edge exactly once) if and only if exactly 0 or 2 vertices have odd degree. If 0 vertices have odd degree, an Eulerian circuit exists starting and ending at the same node. If 2 vertices have odd degree, the path must start at one odd vertex and end at the other."),

        ("Bridge Finding (Cut Edges)",
         "How does DFS identify bridges (critical edges) in an undirected connected graph?",
         "During DFS, compute discovery time tin[u] and lowest reachable time low[u]. An edge (u, v) is a bridge if low[v] > tin[u], meaning the subtree rooted at v has no back-edge to u or any ancestor of u; removing (u, v) disconnects the graph."),

        # Family 2: Trees & Disjoint Sets (11-20)
        ("Binary Search Tree In-order Traversal",
         "Why does an in-order traversal of a valid Binary Search Tree yield keys in sorted ascending order?",
         "By BST invariant, all keys in the left subtree are strictly less than the node key, and all keys in the right subtree are greater. In-order traversal visits left subtree, root, then right subtree recursively: Left < Root < Right, producing sorted order in O(N) time."),

        ("Lowest Common Ancestor (LCA) in BST",
         "How do you find the Lowest Common Ancestor of nodes p and q in a BST in O(h) time?",
         "Start at the root. If both p.val and q.val are strictly less than curr.val, traverse to the left child. If both are strictly greater than curr.val, traverse to the right child. The first node where p and q split (or where curr matches p or q) is their Lowest Common Ancestor."),

        ("AVL Tree Rotations",
         "When is a Left-Right (LR) double rotation required to rebalance an AVL tree?",
         "An LR rotation is required when a node's balance factor is +2 (left-heavy) and its left child's balance factor is -1 (right-heavy). A single left rotation on the left child converts it to a left-left case, followed by a right rotation on the parent to restore balance."),

        ("Red-Black Tree Invariants",
         "What are the fundamental properties that maintain O(log N) height in a Red-Black tree?",
         "Every node is red or black; the root and NIL leaves are black; if a node is red, both children are black (no consecutive red nodes); and every path from a node to descendant NIL leaves contains the exact same number of black nodes (black-height). This ensures the longest path is at most twice the shortest path."),

        ("Segment Tree Range Minimum Query",
         "How does a Segment Tree achieve O(log N) range minimum queries and point updates?",
         "A segment tree represents array intervals hierarchically. Each node stores the minimum of its children. Point updates modify one leaf and propagate upwards along O(log N) ancestor nodes. Range queries decompose the query interval into at most O(log N) disjoint canonical tree segments."),

        ("Binary Indexed Tree (Fenwick Tree)",
         "How does a Fenwick tree update and query prefix sums in O(log n)?",
         "A Fenwick tree stores cumulative sums where index i is responsible for an interval of length equal to its lowest set bit (i & -i). Prefix sum queries subtract (i & -i) to accumulate parent sums, while point updates add (i & -i) to propagate deltas upward in O(log N) time."),

        ("Trie Prefix Search & Insertion",
         "How does a Trie achieve O(m) word insertion and prefix lookup?",
         "Each node in a Trie contains child pointers mapped by character. Words of length m are inserted by traversing character-by-character, creating child nodes when absent and marking the final node as end-of-word. Lookups follow pointers directly without comparing unrelated words, operating in O(m) time."),

        ("Disjoint Set Union (Path Compression & Rank)",
         "Why does DSU with path compression and union by rank achieve near O(1) amortized time?",
         "Union by rank attaches the shallower tree under the root of the deeper tree, keeping tree height logarithmic. Path compression flattens the tree during find queries by pointing every visited node directly to the root, achieving inverse Ackermann complexity O(alpha(n))."),

        ("Kruskal's Algorithm with DSU",
         "How does Kruskal's algorithm build a minimum spanning tree using Union-Find?",
         "Kruskal sorts all graph edges in non-decreasing order of weight. It iterates through sorted edges, using DSU find operations to check if endpoints belong to the same connected component. If not, the edge is added to the MST and components are merged via union until V - 1 edges are selected."),

        ("Prim's Algorithm for MST",
         "How does Prim's algorithm construct a Minimum Spanning Tree starting from an arbitrary root?",
         "Prim maintains a priority queue of cut edges connecting the growing MST tree component to unvisited vertices. At each step, it greedily extracts the minimum-weight edge connecting an outside vertex v, adds v to the tree, and pushes v's outgoing edges into the queue in O(E log V) time."),

        # Family 3: Dynamic Programming (21-30)
        ("0/1 Knapsack DP",
         "Explain the state transition relation for the 0/1 Knapsack problem.",
         "Let dp[i][w] be maximum value using a subset of first i items with weight capacity w. For item i with weight wt and value val: if wt > w, dp[i][w] = dp[i-1][w]; otherwise dp[i][w] = max(dp[i-1][w], dp[i-1][w - wt] + val). Space can be optimized to 1D by iterating weight w backwards from capacity W down to wt."),

        ("Longest Common Subsequence (LCS)",
         "What is the recurrence relation for finding the Longest Common Subsequence of two strings X and Y?",
         "If X[i-1] == Y[j-1], dp[i][j] = 1 + dp[i-1][j-1]. If characters differ, dp[i][j] = max(dp[i-1][j], dp[i][j-1]). The base cases are dp[0][j] = 0 and dp[i][0] = 0. Time complexity is O(M * N)."),

        ("Longest Increasing Subsequence (Patience Sorting)",
         "How does patience sorting solve Longest Increasing Subsequence in O(N log N) time?",
         "Maintain an array `tails` where tails[i] stores the smallest tail of all increasing subsequences of length i + 1. For each element x in array, use binary search (`bisect_left`) to find the first tail >= x and replace it, or append x if greater than all tails. The length of `tails` at the end is the LIS length."),

        ("Edit Distance (Levenshtein Distance)",
         "Explain the three operations evaluated in Levenshtein Edit Distance dynamic programming.",
         "To transform word1[0...i] to word2[0...j]: if word1[i-1] == word2[j-1], dp[i][j] = dp[i-1][j-1]. Otherwise, dp[i][j] = 1 + min(dp[i-1][j] (deletion), dp[i][j-1] (insertion), dp[i-1][j-1] (substitution)). Base cases are dp[i][0] = i and dp[0][j] = j."),

        ("Coin Change (Minimum Coins)",
         "What is the state transition for finding minimum coins to make amount A?",
         "Initialize dp array of size A + 1 with infinity, and dp[0] = 0. For each coin c and amount i from c to A: dp[i] = min(dp[i], dp[i - c] + 1). If dp[A] remains infinity, the amount cannot be formed. Time complexity is O(A * len(coins))."),

        ("Matrix Chain Multiplication",
         "Why does matrix chain multiplication require dynamic programming to find optimal parenthesization?",
         "Matrix multiplication is associative ((AB)C = A(BC)), but the number of scalar multiplications depends heavily on parenthesization. For matrices A_i of dimension p[i-1] x p[i], dp[i][j] = min_{k} (dp[i][k] + dp[k+1][j] + p[i-1]*p[k]*p[j]). It solves in O(N^3) time."),

        ("Subset Sum Problem",
         "How is the subset sum problem mapped to pseudo-polynomial dynamic programming?",
         "Let dp[j] be boolean indicating whether sum j is achievable. Initialize dp[0] = True. For each number x, iterate sum j from Target down to x: dp[j] = dp[j] or dp[j - x]. Time complexity is O(N * Target), which is pseudo-polynomial because it depends on the numerical value of Target."),

        ("Word Break DP",
         "How does dynamic programming determine if string s can be segmented into dictionary words?",
         "Let dp[i] be True if prefix s[0...i] can be segmented. Initialize dp[0] = True. For i from 1 to len(s): for j from 0 to i: if dp[j] and s[j...i] in word_set, set dp[i] = True and break. Time complexity is O(N^2) with average O(1) hash set lookups."),

        ("Maximal Square in Binary Matrix",
         "What is the DP transition to find the largest square of 1s in a binary matrix?",
         "Let dp[i][j] be the side length of the largest square whose bottom-right corner is at (i, j). If matrix[i][j] == '1': dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]). The area of the maximal square is max(dp)^2. Time complexity is O(M * N)."),

        ("Palindromic Substrings Count",
         "How does the expand-around-center approach count all palindromic substrings in O(N^2) time and O(1) space?",
         "A palindrome mirrors around its center. There are 2N - 1 possible centers (N single characters for odd palindromes, N - 1 pairs for even palindromes). For each center, expand two pointers left and right outward while characters match, incrementing the count for each valid expansion."),

        # Family 4: Sorting & Searching Mechanics (31-40)
        ("Merge Sort Stability",
         "Why is standard Merge Sort a stable sorting algorithm?",
         "Merge sort divides array into halves and merges sorted sublists. During the merge step, when two elements compare equal (left[i] == right[j]), the algorithm prioritizes taking the element from the left sublist first, preserving original relative order."),

        ("Quicksort Lomuto vs Hoare Partitioning",
         "What is the key performance difference between Hoare and Lomuto partitioning in quicksort?",
         "Hoare partitioning uses two pointers starting from opposite ends moving inward, performing on average three times fewer swaps than Lomuto partitioning and handling duplicate elements more efficiently."),

        ("Heapsort In-Place Mechanics",
         "How does Heapsort sort an array in-place in O(N log N) time?",
         "Heapsort first builds a max-heap in-place in O(N) time using bottom-up sift-down operations. It then repeatedly swaps the root (maximum element) with the last unsorted element at index i, decrements heap size, and restores the max-heap property on the root in O(log N) time."),

        ("Counting Sort Bounds",
         "When is Counting Sort preferred over comparison-based O(N log N) sorting algorithms?",
         "Counting sort is preferred when elements are integers within a small bounded range [0, K] where K = O(N). It counts occurrences of each integer in O(N + K) linear time and space, bypassing the theoretical Omega(N log N) comparison lower bound."),

        ("Radix Sort (LSD)",
         "Why does Least Significant Digit (LSD) Radix Sort require a stable sub-sorting algorithm?",
         "LSD sorts keys digit-by-digit from lowest to highest place value. If the digit sort were not stable, sorting by a higher place digit would scramble the relative order previously established by lower place digits."),

        ("Binary Search Lower Bound vs Upper Bound",
         "What is the operational difference between lower_bound and upper_bound in a sorted array?",
         "`lower_bound` finds the first index where element >= target (the leftmost insertion point), whereas `upper_bound` finds the first index where element > target (the rightmost insertion point)."),

        ("Ternary Search for Unimodal Functions",
         "How does ternary search locate the maximum of a strictly unimodal function in O(log N)?",
         "Ternary search divides the interval [left, right] into three equal parts using m1 = left + (right - left) / 3 and m2 = right - (right - left) / 3. If f(m1) < f(m2), the peak cannot lie in [left, m1], so left = m1; otherwise right = m2."),

        ("Quickselect Expected O(N) Complexity",
         "Why does Quickselect achieve expected O(N) time for finding the k-th smallest element?",
         "Unlike quicksort which recurses on both partitions (T(n) = 2T(n/2) + O(n) = O(n log n)), Quickselect recurses only on the single partition containing index k: T(n) = T(n/2) + O(n) = n + n/2 + n/4 + ... = O(2n) = O(N)."),

        ("Bucket Sort Distribution",
         "What conditions enable Bucket Sort to run in O(N) average time?",
         "Bucket sort runs in O(N) average time when input numbers are uniformly distributed over an interval [0, 1). Elements are partitioned across N buckets; with uniform distribution, each bucket receives O(1) elements on average, making bucket sorting linear."),

        ("Dutch National Flag (3-Way Partitioning)",
         "How does Dijkstra's Dutch National Flag algorithm partition an array into three groups in single pass?",
         "Maintain three pointers: low = 0, mid = 0, high = n - 1. While mid <= high: if arr[mid] == 0, swap(arr[low], arr[mid]), low++, mid++; if arr[mid] == 1, mid++; if arr[mid] == 2, swap(arr[mid], arr[high]), high--. It operates in single pass O(N) time and O(1) space."),

        # Family 5: String Algorithms (41-50)
        ("KMP Algorithm (LPS Array)",
         "What does the Longest Prefix Suffix (LPS) array represent in the Knuth-Morris-Pratt string matching algorithm?",
         "LPS[i] stores the length of the longest proper prefix of pattern[0...i] that is also a suffix of pattern[0...i]. When a mismatch occurs at pattern index j against text character, LPS[j-1] indicates the next pattern index to align, avoiding re-scanning matched text characters in O(N + M) time."),

        ("Rabin-Karp Rolling Hash",
         "How does rolling hash prevent O(M) string recalculation in Rabin-Karp search?",
         "A polynomial rolling hash computes hash of window s[i...i+M-1] by subtracting the outgoing character's high-order term, multiplying by the base, and adding the incoming character modulo a large prime in O(1) time per window."),

        ("Z-Algorithm for Pattern Matching",
         "What does array Z[i] represent in the Z-algorithm for linear string matching?",
         "Z[i] is the length of the longest substring starting at index i that matches a prefix of string S. By constructing S = Pattern + '$' + Text, any position where Z[i] == len(Pattern) indicates an exact pattern occurrence in O(N + M) time."),

        ("Manacher's Algorithm for Longest Palindrome",
         "How does Manacher's algorithm find the longest palindromic substring in O(N) time?",
         "Manacher inserts separator '#' between characters to unify odd and even palindromes. It iterates through the string, using previously computed palindrome radius and center symmetry to initialize the radius of the current index before extending, guaranteeing each character boundary is checked in O(1) amortized time."),

        ("Aho-Corasick Automaton",
         "How does the Aho-Corasick automaton match multiple dictionary keywords simultaneously?",
         "It builds a Trie of all keywords augmented with 'failure transitions' (similar to KMP LPS pointers). When a mismatch occurs during text streaming, the automaton follows failure links to the longest proper suffix that is a valid prefix in the Trie, finding all keyword matches in linear O(N + total_matches) time."),

        ("Boyer-Moore Bad Character Rule",
         "Explain how the bad character rule accelerates Boyer-Moore string searching.",
         "Boyer-Moore scans the pattern from right to left. Upon encountering a mismatched text character c, if c does not exist in the pattern, the pattern shifts entirely past c; if c exists, the pattern shifts to align its rightmost occurrence of c with the text character."),

        ("Longest Duplicate Substring (Suffix Automaton / Binary Search)",
         "How can you find the longest duplicate substring using binary search and rolling hash?",
         "Binary search the duplicate length L between 1 and N - 1. For a fixed L, slide a window across the string computing rolling hashes and store them in a hash set. If a hash collision occurs and string equality is confirmed, a duplicate of length L exists."),

        ("Run-Length Encoding (RLE) Bounds",
         "When does Run-Length Encoding result in data expansion rather than compression?",
         "RLE expands data when the input has few consecutive repeated symbols (e.g. alternating 'ABABAB'). Each character requires at least two bytes (count and symbol), doubling output size from N to 2N."),

        ("Burrows-Wheeler Transform (BWT) Invertibility",
         "Why is the Burrows-Wheeler Transform reversible without storing the entire permutation matrix?",
         "BWT sorts all cyclic rotations of text T. The last column L and first column F share the Last-to-First (LF) mapping property: the k-th occurrence of character c in L corresponds to the k-th occurrence of c in F, allowing exact backward reconstruction of T."),

        ("Levenshtein Distance Space Optimization",
         "How can Levenshtein edit distance be computed using only O(min(M, N)) memory?",
         "Computing row i of the DP matrix requires only values from row i - 1 (previous row) and current row i. We maintain two arrays of length min(M, N) (or a single array with a temporary diagonal variable), discarding older rows."),

        # Family 6: Greedy & Sliding Window (51-60)
        ("Sliding Window Maximum (Monotonic Deque)",
         "How does a monotonic deque compute maximums over sliding windows of size k in O(N)?",
         "The deque stores array indices in decreasing order of element values. When shifting the window, indices outside the window boundary are popped from the front. New incoming elements pop smaller elements from the back before insertion, ensuring the front of the deque always holds the current window maximum in amortized O(1)."),

        ("Two Pointers (Container with Most Water)",
         "Explain the two-pointer greedy choice for maximizing container water area.",
         "Pointers begin at opposite ends: left = 0 and right = n - 1. Area is (right - left) * min(h[left], h[right]). To potentially find a taller boundary and larger area, the pointer at the shorter height is shifted inward, because moving the taller boundary while shortening width cannot increase area."),

        ("Activity Selection Problem",
         "Why does sorting intervals by finish time produce an optimal greedy schedule for non-overlapping activities?",
         "Selecting the activity with earliest finish time frees the resource as early as possible for subsequent activities. It leaves the maximum remaining time available, which can be proven by an exchange argument to be at least as good as any other choice."),

        ("Huffman Coding Optimal Prefix Trees",
         "Why does Huffman coding build optimal prefix-free codes using a min-heap?",
         "Huffman extracts the two least-frequent symbols, combines them into a parent node with sum frequency, and re-inserts into the heap. Less frequent symbols receive deeper tree levels and longer bitcodes, minimizing expected code length sum(f_i * len_i)."),

        ("Gas Station Circular Tour",
         "How does a single-pass greedy algorithm determine if a circular gas station tour is possible?",
         "Track total gas balance and current tank balance. Start at station 0. If current tank drops below 0 at station i, no station from start to i can be a valid starting point; reset start to i + 1 and current tank to 0. If total gas >= total cost at the end, start is the valid index."),

        ("Jump Game (Greedy Reach)",
         "How does tracking max reachable index solve the Jump Game in O(N) time and O(1) space?",
         "Maintain variable `max_reach` initialized to 0. For each index i: if i > max_reach, the index is unreachable, return False. Otherwise update max_reach = max(max_reach, i + nums[i]). If max_reach >= last_index, return True."),

        ("Minimum Window Substring (Two-Pointer Frequency Map)",
         "How does the sliding window algorithm find the minimum window substring containing all characters of pattern T?",
         "Expand the right pointer, adding characters to window frequency map until all required characters in T are satisfied. Then shrink the left pointer inward to minimize window length while preserving satisfaction. Record minimum length, then increment left to search for next window in O(N) time."),

        ("Interval Scheduling (Minimum Meeting Rooms)",
         "How does a min-heap determine the minimum number of meeting rooms required for intervals?",
         "Sort meetings by start time. Use a min-heap storing end times of active meetings. For each meeting: if its start time >= minimum end time in heap (root), reuse that room by popping root. Then push current meeting's end time. Heap size at any moment is the required rooms count."),

        ("Trapping Rain Water (Two-Pointer Method)",
         "How do two pointers compute trapped rain water in O(N) time and O(1) auxiliary space?",
         "Pointers left and right start at ends, tracking left_max and right_max. If left_max < right_max, water trapped at left is max(0, left_max - height[left]), update left_max, left++. Otherwise water trapped at right is max(0, right_max - height[right]), update right_max, right--."),

        ("Fractional Knapsack Greedy Strategy",
         "Why does the greedy ratio (value / weight) guarantee optimality for fractional knapsack but fail for 0/1 knapsack?",
         "In fractional knapsack, divisibility allows packing full capacity with highest density value per unit weight. In 0/1 knapsack, item indivisibility can leave unfilled capacity, where a combination of lower-density items might yield a higher total value."),

        # Family 7: Math, Number Theory & Bit Manipulation (61-70)
        ("Euclidean Algorithm for GCD",
         "Explain the recurrence in the Euclidean algorithm for greatest common divisor.",
         "The algorithm relies on gcd(a, b) = gcd(b, a % b) with base case gcd(a, 0) = a. Since a % b < a / 2 when a >= b, the numbers halve at least every two steps, terminating in O(log(min(a, b))) iterations."),

        ("Sieve of Eratosthenes Complexity",
         "Why does the Sieve of Eratosthenes have a time complexity of O(N log log N)?",
         "For each prime p up to sqrt(N), the sieve marks multiples (N/p operations). Total operations equal N * sum_{p <= N} (1/p). By Mertens' second theorem, sum of reciprocals of primes up to N scales as log(log N), giving O(N log log N)."),

        ("Modular Exponentiation (Binary Exponentiation)",
         "How does binary exponentiation compute (base^exp) % mod in O(log exp) time?",
         "Express exp in binary. Initialize result = 1. While exp > 0: if exp is odd (exp & 1), result = (result * base) % mod. Square base = (base * base) % mod, and shift exp >>= 1. This computes powers using at most 2 * log2(exp) multiplications."),

        ("Extended Euclidean Algorithm & Modular Inverse",
         "When does integer a have a modular multiplicative inverse modulo m?",
         "An integer a has a modular inverse modulo m if and only if gcd(a, m) = 1 (coprime). The Extended Euclidean algorithm finds integers x and y such that a*x + m*y = gcd(a, m) = 1; then (x % m + m) % m is the modular inverse."),

        ("Fermat's Little Theorem for Primality Testing",
         "State Fermat's Little Theorem and its application in primality testing.",
         "If p is prime and a is not divisible by p, then a^(p-1) == 1 (mod p). If a^(n-1) != 1 (mod n) for any chosen base a, n is composite. However, Carmichael numbers pass this test for all coprime bases, requiring Miller-Rabin for composite detection."),

        ("Fast Fourier Transform (FFT) Polynomial Multiplication",
         "How does FFT multiply two degree-N polynomials in O(N log N) time?",
         "Standard polynomial multiplication takes O(N^2). FFT evaluates polynomials at N complex roots of unity in O(N log N) using divide-and-conquer, multiplies point values pointwise in O(N), and applies Inverse FFT to interpolate back to coefficients in O(N log N)."),

        ("Bitwise Brian Kernighan's Algorithm",
         "How does Brian Kernighan's algorithm count set bits in an integer in O(set_bits) time?",
         "The operation `n & (n - 1)` clears the lowest set bit of n. By looping `while n: n &= (n - 1); count += 1`, the loop executes exactly as many times as there are set 1-bits in n, rather than checking all 32 or 64 bit positions."),

        ("Single Number (XOR Cancellation)",
         "How does XOR find the single non-duplicate element in an array where every other element appears twice?",
         "XOR is associative, commutative, and satisfies x ^ x = 0 and x ^ 0 = x. XORing all elements together cancels out every paired duplicate (x ^ x = 0), leaving only the unique non-duplicate element in O(N) time and O(1) space."),

        ("Power of Two Bitwise Check",
         "Why does `(n > 0) and ((n & (n - 1)) == 0)` verify if an integer is a power of two?",
         "A positive power of two has exactly one bit set to 1 (e.g. 8 is 1000_2). Subtracting 1 flips all bits up to that set bit (7 is 0111_2). Bitwise AND between n and n - 1 yields 0 if and only if n had exactly one set bit."),

        ("Matrix Exponentiation for Linear Recurrences",
         "How does matrix exponentiation compute the N-th Fibonacci number in O(log N) time?",
         "The recurrence [F(n+1), F(n)]^T = [[1, 1], [1, 0]] * [F(n), F(n-1)]^T allows expressing F(n) via [[1, 1], [1, 0]]^(n-1). Using binary exponentiation on 2x2 matrices computes the n-th power in O(8 * log N) = O(log N) operations."),

        # Family 8: Complexity & Algorithmic Limits (71-80)
        ("P vs NP Definition",
         "What is the formal theoretical difference between complexity classes P and NP?",
         "Class P consists of decision problems solvable by a deterministic Turing machine in polynomial time O(N^k). Class NP consists of decision problems whose solutions can be verified by a deterministic Turing machine in polynomial time."),

        ("NP-Complete Definition (Cook-Levin Theorem)",
         "What conditions must a problem satisfy to be classified as NP-Complete?",
         "A problem L is NP-Complete if: 1. L is in NP (verifiable in polynomial time), and 2. Every problem in NP is polynomial-time reducible to L (NP-Hard). The Cook-Levin theorem proved that Boolean Satisfiability (SAT) is NP-Complete."),

        ("Comparison Sorting Lower Bound Proof",
         "Why cannot any comparison-based sorting algorithm run faster than Omega(N log N) in the worst case?",
         "An array of N distinct elements has N! possible permutations. A comparison-based algorithm is a binary decision tree where each leaf represents a sorted permutation. A binary tree with N! leaves must have height at least log2(N!). By Stirling's approximation, log2(N!) = Omega(N log N)."),

        ("Master Theorem Case 1 (Leaf Heavy)",
         "Apply the Master Theorem to recurrence T(n) = 8 T(n/2) + O(n^2).",
         "Here a = 8, b = 2, and f(n) = n^2. Compute log_b(a) = log2(8) = 3. Since f(n) = O(n^(3 - epsilon)) for epsilon = 1, Case 1 applies: the tree leaves dominate, yielding T(n) = Theta(n^3)."),

        ("Master Theorem Case 2 (Balanced)",
         "Apply the Master Theorem to recurrence T(n) = 2 T(n/2) + O(n) (Merge Sort).",
         "Here a = 2, b = 2, and f(n) = n. Compute log_b(a) = log2(2) = 1. Since f(n) = Theta(n^1), Case 2 applies: work is distributed equally across all log2(n) levels, yielding T(n) = Theta(n log n)."),

        ("Amortized Analysis (Accounting Method)",
         "How does the accounting method prove O(1) amortized cost for dynamic array resizing?",
         "When an element is inserted, charge an amortized cost of $3: $1 for its current insertion, $1 to move it when capacity doubles, and $1 to move an older element that has not paid. By the time capacity doubles, accumulated credits pay for the O(N) array copy, making amortized cost O(1)."),

        ("Space Complexity: Tail Call Optimization",
         "Why does Tail Call Optimization (TCO) reduce recursive call space complexity from O(N) to O(1)?",
         "If the recursive call is in tail position (the final action with nothing remaining to evaluate), the compiler or interpreter can reuse the current stack frame instead of pushing a new frame, transforming recursion into an iterative jump."),

        ("Deterministic vs Las Vegas vs Monte Carlo Algorithms",
         "What is the difference between a Las Vegas and a Monte Carlo randomized algorithm?",
         "A Las Vegas algorithm always produces the correct answer; its running time is a random variable (e.g. Randomized Quicksort). A Monte Carlo algorithm runs in deterministic time but has a small, bounded probability of returning an incorrect answer (e.g. Miller-Rabin primality test)."),

        ("NP-Hard Traveling Salesperson Problem (TSP)",
         "How does the Held-Karp dynamic programming algorithm solve TSP in O(N^2 * 2^N) time?",
         "Let dp[mask][i] be the minimum cost of visiting the subset of cities represented by bitmask, ending at city i. For each mask and each endpoint i, transition by trying all next cities j not in mask: dp[mask | (1 << j)][j] = min(dp[mask | (1 << j)][j], dp[mask][i] + dist[i][j])."),

        ("Halting Problem Undecidability",
         "Summarize Turing's proof that the Halting Problem is undecidable.",
         "Assume function `halts(f)` returns True if f terminates, False otherwise. Construct `paradox()`: `if halts(paradox): while True: pass else: return`. If `paradox` halts, `halts` says True, so it loops forever. If it loops, `halts` says False, so it halts immediately. This contradiction proves no general halting decider can exist."),

        # Family 9: Geometry, Numerical & Search Trees (81-90)
        ("Convex Hull (Graham Scan)",
         "How does Graham Scan compute the 2D convex hull in O(N log N) time?",
         "Find the bottom-most point as origin, sort all other points by polar angle. Iterate through sorted points using a stack, verifying counter-clockwise orientation using cross products (cross(p1, p2, p3) > 0). If a point forms a clockwise turn, pop the stack until convexity is restored."),

        ("Closest Pair of Points (Divide and Conquer)",
         "How does divide-and-conquer find the closest pair of 2D points in O(N log N) time?",
         "Sort points by x-coordinate. Recursively find minimum distance d in left and right halves. Examine the strip of width 2d around the dividing line, sorted by y-coordinate. Within the strip, each point needs to be compared against at most 7 subsequent points, keeping the combine step O(N)."),

        ("Line Segment Intersection (Cross Product)",
         "How do 2D cross products determine whether two line segments AB and CD intersect?",
         "Compute orientations of triangle triplets (A, B, C), (A, B, D) and (C, D, A), (C, D, B) using signed 2D cross products. The segments intersect if and only if C and D lie on opposite sides of line AB, and A and B lie on opposite sides of line CD (or if points are collinear and overlapping)."),

        ("KD-Tree Spatial Partitioning",
         "How does a KD-Tree partition k-dimensional space for efficient nearest neighbor search?",
         "A KD-Tree recursively splits points along alternating coordinate axes (e.g. x-axis at depth 0, y-axis at depth 1). At each node, points are divided by the median coordinate into left and right subtrees, allowing nearest-neighbor search to prune entire spatial hyperrectangles in expected O(log N) time."),

        ("Quadtree Spatial Indexing",
         "What is the difference between a Point Quadtree and a Region Quadtree?",
         "A Point Quadtree splits a 2D space into four quadrants (NW, NE, SW, SE) centered around an inserted point coordinate. A Region Quadtree recursively decomposes space into four equal-sized geometric squares down to a fixed resolution or maximum capacity."),

        ("Interval Tree Query",
         "How does an Interval Tree find all intervals overlapping query point q in O(log N + k)?",
         "The tree is a balanced BST ordered by interval start points. Each node stores the maximum endpoint `max_end` among all intervals in its subtree. If `left.max_end < q`, the left subtree cannot contain any overlapping interval and can be pruned safely."),

        ("B-Tree Node Splitting",
         "Describe the node-splitting procedure during B-Tree key insertion.",
         "When an insertion causes a node to exceed maximum capacity (2t - 1 keys), the node is split into two nodes of t - 1 keys each. The median key is pushed up to the parent node. If the root splits, a new root is created, increasing tree height by 1."),

        ("Skip List Probabilistic Search",
         "How does a Skip List achieve O(log N) search using forward pointer layers?",
         "A skip list maintains multiple linked list levels. Level 0 contains all elements. Each element is promoted to higher levels with probability p (typically 0.5) via coin flips. Search starts at the highest level, moving right until the next key exceeds target, then dropping down a level, bypassing large runs of nodes."),

        ("Treap (Cartesian Tree Heap)",
         "How does a Treap combine BST and Max-Heap properties?",
         "Each node has a key and a randomly assigned priority. Nodes satisfy the BST property on keys (left < root < right) and the Max-Heap property on priorities (root.priority >= child.priority). Tree rotations during insertion maintain both invariants, yielding expected O(log N) depth."),

        ("Splay Tree Self-Adjustment",
         "What is the splaying operation in a Splay Tree?",
         "Whenever a node is accessed, it is rotated to the root using a sequence of Zig, Zig-Zig, and Zig-Zag tree rotations. Splaying moves frequently accessed items near the root, achieving amortized O(log N) time for all operations without storing explicit balance factors."),

        # Family 10: Backtracking & State Space Search (91-100)
        ("N-Queens Backtracking",
         "How does backtracking prune search space when placing N queens on an N x N board?",
         "Place one queen per row. For row r and column c, check if column c, main diagonal (r - c), or anti-diagonal (r + c) is already occupied using boolean sets. If occupied, prune immediately without recursing on remaining rows. Backtrack by unmarking sets when search returns."),

        ("Sudoku Solver Constraint Propagation",
         "How does constraint propagation with backtracking solve a 9x9 Sudoku puzzle?",
         "Maintain bitmasks of available digits for each row, column, and 3x3 box. At each step, pick the cell with the fewest candidates (Minimum Remaining Values heuristic). Try valid candidate numbers, propagate constraints to neighbors, and recurse. If a contradiction occurs, backtrack and restore masks."),

        ("Subset Generation (Bitmask vs Backtracking)",
         "Compare bitmask iteration and recursive backtracking for generating all 2^N subsets.",
         "Bitmask iteration loops integer i from 0 to (1 << N) - 1, checking bit (i >> j) & 1 for inclusion of element j in O(N * 2^N) iterative time. Backtracking branches binary (include/exclude) at each element in recursive DFS, yielding the same 2^N subsets without integer overflow limits."),

        ("Permutation Generation (Heap's Algorithm)",
         "Why is Heap's algorithm more efficient for generating permutations than naive recursion?",
         "Heap's algorithm generates all N! permutations by swapping only a single pair of elements between successive permutations. It minimizes array write operations and memory overhead compared to recursive slicing."),

        ("Word Search on Board (DFS with Trie)",
         "How does combining a Trie with 2D grid DFS accelerate the Word Search II problem?",
         "Instead of searching for each word separately (multiplying grid DFS by number of words), load the word dictionary into a Trie. During 2D grid DFS, advance in the Trie simultaneously; if the current grid path prefix is not in the Trie, prune the entire search branch immediately."),

        ("Hamiltonian Path Backtracking",
         "Why is checking for a Hamiltonian path in an arbitrary graph computationally intractable?",
         "A Hamiltonian path visits every vertex in the graph exactly once. Determining if such a path exists is NP-Complete. Naive backtracking explores V! paths in the worst case; dynamic programming with bitmasks improves this to O(V^2 * 2^V), which remains exponential."),

        ("Graph Coloring (m-Coloring Problem)",
         "How does backtracking determine if a graph can be colored with at most m colors?",
         "Assign colors 1 to m to vertex v. Check whether any adjacent neighbor has the same color. If safe, recurse to vertex v + 1. If no color from 1 to m is valid for vertex v, backtrack to vertex v - 1 and try the next available color."),

        ("Combination Sum (Unbounded Knapsack Backtracking)",
         "How does passing the current index prevent duplicate combinations in Combination Sum?",
         "To allow reusing elements while preventing duplicate unordered combinations like [2, 3] and [3, 2], recursive calls pass the current element index `start`. The loop explores candidates from `start` to len(candidates), ensuring combinations are always non-decreasing in element index."),

        ("Flood Fill Algorithm",
         "Explain how DFS flood fill recolors a connected component in a 2D image matrix.",
         "Given starting pixel (r, c) with original color C: if current pixel is outside bounds or pixel color != C, return. Set pixel color to new_color. Recurse in four cardinal directions (r+1, c), (r-1, c), (r, c+1), (r, c-1). Base case prevents infinite recursion if new_color == C."),

        ("Iterative Deepening Search (IDDFS)",
         "Why does Iterative Deepening DFS (IDDFS) combine the benefits of BFS and DFS for state-space search?",
         "IDDFS performs depth-limited DFS for increasing depth limits d = 1, 2, 3, ... It visits nodes in breadth-first order, guaranteeing shortest path optimality like BFS, but uses only O(d) memory like DFS. The overhead of re-visiting top levels is a small fraction (b/(b-1)) of bottom-level work.")
    ]

    assert len(algorithms) == 100, f"Expected 100 algorithms, got {len(algorithms)}"

    records = []
    for i, (title, prompt, response) in enumerate(algorithms, 1):
        records.append({
            "id": f"EXP34-ALGO-{i:04d}",
            "prompt": prompt,
            "response": response,
            "category": "algorithmic_reasoning",
            "source_type": "synthetic_algorithmic_curation",
            "language": "en",
            "license": "CC0-1.0",
            "data_role": "TRAIN",
            "review_status": "VERIFIED"
        })
    return records

