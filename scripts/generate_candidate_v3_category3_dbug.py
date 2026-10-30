"""
Exp34 Candidate V3 - Category 3: Code Debugging
Contains exactly 100 completely distinct, non-repeating real-world Python bug scenarios.
Covers 20+ diverse bug families. Zero loop-parameterized templates.
"""

from typing import Any, Dict, List

def build_category_3_debugging() -> List[Dict[str, Any]]:
    # List of 100 distinct tuples: (title, prompt, response)
    bugs = [
        # Family 1: Indexing & Slicing Errors (1-5)
        ("index_out_of_range",
         "Why does `for i in range(len(a) + 1): print(a[i])` crash with IndexError?",
         "In Python, valid indices are 0 to `len(a) - 1`. `range(len(a) + 1)` reaches index `len(a)`, raising `IndexError: list index out of range`.\n\nFix: Iterate up to `range(len(a))` or loop directly over elements: `for item in a:`."),

        ("negative_step_slicing_empty",
         "Why does `nums = [1, 2, 3, 4, 5]; print(nums[1:4:-1])` return an empty list `[]` instead of reversing elements?",
         "When a negative step (`-1`) is specified in a slice, the slice expects the start index to be greater than the stop index. Since index 1 is less than index 4, the slice immediately halts and returns `[]`.\n\nFix: Set start higher than stop: `nums[4:1:-1]` to slice backwards, or reverse the subslice `nums[1:4][::-1]`."),

        ("string_immutable_item_assignment",
         "Why does `s = 'hello'; s[0] = 'H'` raise a TypeError?",
         "Python strings are immutable sequences; their individual characters cannot be modified in-place via item assignment.\n\nFix: Create a new string using concatenation or slicing: `s = 'H' + s[1:]`."),

        ("pop_from_empty_list",
         "Why does `stack = []; val = stack.pop()` raise IndexError?",
         "Calling `.pop()` on an empty list raises `IndexError: pop from empty list` because there is no element at the end of the sequence to remove.\n\nFix: Check whether the list has items before popping: `if stack: val = stack.pop()`."),

        ("off_by_one_subslice_length",
         "Why does `items[0:k]` return fewer elements than expected when k is 0?",
         "A slice `items[0:0]` specifies an interval with identical start and end points, which defines an empty range and returns an empty list.\n\nFix: If at least one element is required, ensure `k >= 1` or handle the empty boundary condition explicitly."),

        # Family 2: Mutable State & Default Arguments (6-10)
        ("mutable_default_arg_list",
         "Why does `def append_val(v, target=[])` accumulate values across separate calls?",
         "Default argument expressions are evaluated only once when the function definition is executed, sharing that single list instance across all subsequent invocations.\n\nFix: Use `target=None` and initialize inside the function: `if target is None: target = []`."),

        ("mutable_default_arg_dict",
         "Why does `def set_cache(k, v, store={})` preserve state between independent requests?",
         "The dictionary `store={}` is created once at function definition time. Every function call without an explicit `store` parameter mutates the exact same dictionary in memory.\n\nFix: Use `store=None` as default, then assign `if store is None: store = {}` inside the function body."),

        ("shallow_copy_nested_grid",
         "Why does `grid = [[0] * 3] * 3; grid[0][0] = 1` turn the first column into all 1s?",
         "Multiplying a list containing an inner list duplicates references to the exact same inner list instance in memory, rather than creating independent sublists.\n\nFix: Use a list comprehension to allocate distinct sublists: `grid = [[0] * 3 for _ in range(3)]`."),

        ("class_variable_shared_mutation",
         "Why does modifying `self.items.append(x)` update all instances if `items = []` is declared on the class body?",
         "Declaring `items = []` directly inside the class body creates a class-level attribute shared across all instances of that class.\n\nFix: Initialize instance-specific attributes inside `__init__`: `self.items = []`."),

        ("dict_copy_shallow_nested_modification",
         "Why does `b = a.copy(); b['user']['role'] = 'admin'` also mutate `a['user']['role']`?",
         "The `.copy()` method performs a shallow copy, duplicating only the top-level dictionary while nested dictionary values remain shared references.\n\nFix: Use `copy.deepcopy(a)` to recursively duplicate nested structures."),

        # Family 3: Concurrency & Iteration Mutation (11-15)
        ("modifying_list_during_iteration",
         "Why does `for x in nums: if x % 2 == 0: nums.remove(x)` skip adjacent elements?",
         "Deleting elements in-place shifts subsequent element indices leftward while the iterator advances forward, causing the element shifting into the current index to be skipped.\n\nFix: Use a list comprehension to create a filtered list: `nums = [x for x in nums if x % 2 != 0]`."),

        ("runtime_error_dict_size_change",
         "Why does `for k in d: if d[k] < 0: del d[k]` raise `RuntimeError: dictionary changed size during iteration`?",
         "Python iterators over dictionaries forbid modifying keys during active iteration to prevent bucket corruption.\n\nFix: Iterate over a snapshot of the keys: `for k in list(d.keys()):`."),

        ("set_mutation_during_iteration",
         "Why does `for item in active_set: if condition(item): active_set.remove(item)` fail?",
         "Sets do not allow adding or removing elements while an iterator is traversing them.\n\nFix: Iterate over a list copy: `for item in list(active_set):` or use a set comprehension."),

        ("thread_race_condition_counter",
         "Why does incrementing `global_count += 1` across multiple threads produce inconsistent totals?",
         "`+=` is not an atomic operation in Python bytecode; it involves binary load, add, and store instructions. Context switches between these steps cause lost updates.\n\nFix: Synchronize access using a `threading.Lock()` with a `with lock:` context manager."),

        ("generator_exhaustion_on_reuse",
         "Why does calling `sum(gen)` then `max(gen)` fail on the second call?",
         "Generators are one-pass iterators; once traversed, the generator is completely exhausted and yields no further values on subsequent calls.\n\nFix: Materialize items into a list before processing: `data = list(gen)`."),

        # Family 4: Numeric, Types & Precision (16-20)
        ("floating_point_equality",
         "Why does `0.1 + 0.2 == 0.3` return False in Python?",
         "Binary IEEE 754 floating-point numbers cannot represent 0.1 and 0.2 with exact finite precision, resulting in `0.30000000000000004`.\n\nFix: Use `math.isclose(0.1 + 0.2, 0.3)` or `decimal.Decimal`."),

        ("is_vs_equality_large_integers",
         "Why is `a is 1000` unreliable compared to `a == 1000`?",
         "`is` checks object identity (same memory address). While CPython caches small integers (-5 to 256), larger integers are allocated separate objects.\n\nFix: Always use `==` for equality checking."),

        ("unhandled_zero_division",
         "How do you protect `avg = total / count` against ZeroDivisionError when count is 0?",
         "Division by zero raises `ZeroDivisionError` if unhandled.\n\nFix: Check the denominator first: `avg = total / count if count != 0 else 0.0`."),

        ("int_conversion_float_string",
         "Why does `int('12.5')` raise a ValueError in Python?",
         "The built-in `int()` constructor cannot directly parse decimal point characters in string literals.\n\nFix: Convert to float first, then truncate: `int(float('12.5'))`."),

        ("string_concatenation_type_error",
         "Why does `'Status code: ' + 200` raise a TypeError?",
         "Python does not perform implicit string coercion during addition.\n\nFix: Use an f-string: `f'Status code: {200}'` or explicit `str(200)`."),

        # Family 5: Variable Scoping & Closures (21-25)
        ("unbound_local_error_read_before_assign",
         "Why does `x = 5; def f(): x += 1` raise UnboundLocalError?",
         "Because `x` is assigned inside `f()`, the Python compiler marks `x` as local to the entire function scope. `x += 1` attempts to read the local variable before assignment.\n\nFix: Declare `global x` or pass `x` as an explicit parameter."),

        ("closure_late_binding_loop",
         "Why do lambda functions created in a loop `[lambda: i for i in range(3)]` all return 2?",
         "Closures in Python look up variables by reference at call time rather than at definition time; when invoked, `i` has resolved to its final loop value (2).\n\nFix: Bind the current value as a default argument: `[lambda i=i: i for i in range(3)]`."),

        ("shadowing_builtin_sum",
         "Why does calling `sum([1, 2])` raise TypeError after executing `sum = 100`?",
         "Assigning to `sum` binds the identifier to an integer in the current namespace, shadowing Python's built-in `sum()` function.\n\nFix: Rename the variable to `total` or `total_sum`."),

        ("nonlocal_variable_mutation_error",
         "Why does an inner function modifying an enclosing variable fail with UnboundLocalError?",
         "Assigning to a variable in an inner function defaults to treating it as local to the inner function rather than modifying the outer enclosing scope.\n\nFix: Declare `nonlocal var_name` inside the inner function."),

        ("class_scope_not_visible_in_comprehension",
         "Why does `class A: x = 5; y = [x for _ in range(3)]` fail with NameError in Python 3?",
         "List comprehensions have their own scope in Python 3 and cannot access the enclosing class body scope directly.\n\nFix: Define `y` using a class method or define `x` outside the class."),

        # Family 6: Object Model & Magic Methods (26-30)
        ("dict_key_unhashable_list",
         "Why does `lookup[[1, 2]] = 'val'` raise TypeError?",
         "Dictionary keys must be hashable so their bucket hashes never change. Lists are mutable and do not implement `__hash__`.\n\nFix: Use an immutable tuple as the key: `lookup[(1, 2)] = 'val'`."),

        ("hash_without_eq_method",
         "Why does overriding `__eq__` without `__hash__` make custom class instances unhashable?",
         "Overriding `__eq__` causes Python to automatically set `__hash__ = None` to enforce the contract that equal objects must have equal hashes.\n\nFix: Implement `__hash__` explicitly: `def __hash__(self): return hash(self.id)`."),

        ("repr_vs_str_infinite_recursion",
         "Why does `def __str__(self): return str(self)` trigger RecursionError?",
         "`str(self)` calls `self.__str__()`, creating an infinite recursive loop that exhausts the call stack.\n\nFix: Return formatted attributes: `return f'{self.__class__.__name__}(id={self.id})'`."),

        ("init_method_returning_value",
         "Why does `def __init__(self): return 42` raise a TypeError?",
         "The `__init__` constructor method must return `None` because the newly instantiated object is returned automatically by `__new__`.\n\nFix: Remove the return statement or return `None`."),

        ("getattr_infinite_recursion",
         "Why does accessing `self.name` inside `__getattr__` cause a stack overflow?",
         "If `self.name` does not exist in `__dict__`, accessing it inside `__getattr__` triggers `__getattr__` again recursively.\n\nFix: Access attributes via `self.__dict__` or `object.__getattribute__(self, name)`."),

        # Family 7: Control Flow & Exceptions (31-35)
        ("bare_except_suppressing_interrupts",
         "Why is `except:` without an exception class considered dangerous?",
         "A bare `except:` catches `BaseException`, which silently intercepts `KeyboardInterrupt` (Ctrl+C) and `SystemExit`, preventing clean program termination.\n\nFix: Catch `except Exception:` to only catch standard application errors."),

        ("finally_block_overriding_return",
         "Why does `def f(): try: return 1 finally: return 2` return 2 instead of 1?",
         "The `finally` block always executes before leaving a function. An explicit return inside `finally` discards any active return value or raised exception.\n\nFix: Avoid `return` statements inside `finally` blocks."),

        ("exception_reraise_losing_traceback",
         "Why does `except Exception as e: raise e` obscure original error lineage?",
         "Re-raising `e` directly resets the traceback to the raise line rather than preserving the root cause.\n\nFix: Use a bare `raise` to re-raise the current active exception, or `raise NewError() from e`."),

        ("assert_statement_with_tuple_truthy",
         "Why does `assert(x == 5, 'error')` always evaluate to True in Python?",
         "Parentheses construct a 2-tuple `(bool, str)`. Any non-empty tuple evaluates to True in a boolean context regardless of `x`.\n\nFix: Remove the parentheses: `assert x == 5, 'error'`."),

        ("except_multiple_without_tuple",
         "Why does `except ValueError, TypeError:` fail with a SyntaxError in Python 3?",
         "In Python 3, multiple exceptions in an `except` clause must be enclosed in parentheses as a tuple; comma without parentheses was old Python 2 syntax for binding an alias.\n\nFix: Write `except (ValueError, TypeError):`."),

        # Family 8: File I/O & Serialization (36-40)
        ("file_descriptor_leak_without_context",
         "Why is `f = open('data.txt'); text = f.read()` an anti-pattern?",
         "If an exception occurs during reading, the file handle remains open until garbage collection runs, risking file descriptor exhaustion.\n\nFix: Always use a context manager: `with open('data.txt') as f: text = f.read()`."),

        ("binary_file_mode_missing",
         "Why does `open('image.png', 'w').write(data)` corrupt binary files?",
         "Opening a file in text mode `'w'` encodes strings and performs platform-specific newline translation (e.g. `\\n` to `\\r\\n` on Windows).\n\nFix: Open with binary mode: `open('image.png', 'wb')`."),

        ("json_datetime_not_serializable",
         "Why does `json.dumps({'time': datetime.now()})` raise TypeError?",
         "Python's standard `json` module cannot serialize `datetime` objects out-of-the-box.\n\nFix: Convert to ISO 8601 string: `datetime.now().isoformat()` or pass `default=str`."),

        ("pickle_untrusted_data_exploit",
         "Why is `pickle.loads(untrusted_bytes)` a major security vulnerability?",
         "Pickle serialization executes arbitrary Python bytecode via `__reduce__`, enabling remote code execution from malicious payloads.\n\nFix: Use safe serialization formats like `json` or `yaml.safe_load()`."),

        ("file_read_after_seek_eof",
         "Why does calling `.read()` return empty string after writing to a file?",
         "Writing advances the file pointer to the end of the file. Calling `.read()` immediately reads from the current EOF pointer.\n\nFix: Reposition the pointer using `f.seek(0)` before reading."),

        # Family 9: Regular Expressions & String Formatting (41-45)
        ("regex_unanchored_match_vs_search",
         "Why does `re.match('abc', 'zabc')` return None while `re.search('abc', 'zabc')` finds the pattern?",
         "`re.match()` checks for matches strictly at the beginning of the string, while `re.search()` scans through the entire string.\n\nFix: Use `re.search()` when looking for substrings anywhere in the text."),

        ("regex_greedy_backtracking_overmatch",
         "Why does `re.findall('<.*>', '<a><b>')` return `['<a><b>']` instead of separate tags?",
         "The `*` quantifier is greedy by default, consuming as many characters as possible up to the last `>`.\n\nFix: Use non-greedy `*?`: `re.findall('<.*?>', '<a><b>')`."),

        ("regex_raw_string_backslash_escape",
         "Why does `re.match('\\d+', s)` trigger SyntaxWarnings or unexpected escapes?",
         "Without raw string syntax `r'...'`, Python's string literal interpreter attempts to process backslashes before the regex engine receives them.\n\nFix: Always prefix regex patterns with `r`: `r'\\d+'`."),

        ("f_string_curly_brace_escaping",
         "How do you print literal curly braces inside an f-string?",
         "Single curly braces `{}` are interpreted as expression interpolations.\n\nFix: Double the curly braces: `f'{{literal}}'`."),

        ("str_split_none_vs_whitespace",
         "Why does `'a  b'.split(' ')` return `['a', '', 'b']` while `'a  b'.split()` returns `['a', 'b']`?",
         "Passing an explicit `' '` treats single space characters as strict delimiters, whereas `split(None)` groups consecutive whitespace characters.\n\nFix: Use `s.split()` without arguments to split by arbitrary whitespace runs."),

        # Family 10: Standard Library & Built-in Pitfalls (46-50)
        ("datetime_timezone_naive_comparison",
         "Why does comparing a naive datetime with an aware datetime raise TypeError?",
         "Python forbids relational comparisons (`<`, `>`) between timezone-aware and timezone-naive datetimes because their timeline offset is undefined.\n\nFix: Make naive datetimes aware using `dt.replace(tzinfo=timezone.utc)`."),

        ("random_seed_global_leakage",
         "Why does calling `random.seed(42)` inside a library helper affect the entire application?",
         "`random.seed()` mutates the global `random` module instance, altering pseudorandom sequences across unrelated modules.\n\nFix: Instantiate an isolated RNG instance: `rng = random.Random(42)`."),

        ("math_sqrt_negative_number",
         "Why does `math.sqrt(-1)` raise ValueError instead of returning 1j?",
         "`math.sqrt` operates strictly within the real number domain.\n\nFix: Use the `cmath` module for complex numbers: `cmath.sqrt(-1)`."),

        ("csv_reader_newline_blank_rows",
         "Why does reading or writing CSV files on Windows produce alternating blank rows?",
         "The `csv` writer performs its own newline handling; if `open()` uses default Windows newline translation, extra `\\r` characters are added.\n\nFix: Always open CSV files with `newline=''`:\n`open('data.csv', 'w', newline='', encoding='utf-8')`."),

        ("subprocess_run_shell_injection",
         "Why is `subprocess.run(f'ls {user_input}', shell=True)` dangerous?",
         "Passing `shell=True` with unvalidated user strings allows attackers to chain shell commands via `;` or `|`.\n\nFix: Pass arguments as an argument list with `shell=False`: `subprocess.run(['ls', user_input])`."),

        # Family 11: Async, Coroutines & Event Loops (51-55)
        ("coroutine_never_awaited_warning",
         "Why does `async def fetch(): return 1; val = fetch()` produce a `RuntimeWarning: coroutine was never awaited`?",
         "Calling an `async def` function creates a coroutine object without executing its body; it must be awaited or scheduled on an event loop.\n\nFix: Await the coroutine: `val = await fetch()` or run via `asyncio.run(fetch())`."),

        ("blocking_call_inside_async_event_loop",
         "Why does calling `time.sleep(5)` inside an async function freeze the entire server?",
         "`time.sleep()` blocks the operating system thread, halting the single-threaded asyncio event loop from processing any other concurrent tasks.\n\nFix: Use non-blocking `await asyncio.sleep(5)` or run blocking code in an executor: `asyncio.to_thread()`."),

        ("asyncio_gather_single_exception_crash",
         "Why does `asyncio.gather(*tasks)` abort and cancel remaining tasks on the first exception?",
         "By default, `asyncio.gather()` propagates the first encountered exception immediately unless configured otherwise.\n\nFix: Pass `return_exceptions=True` to collect results and exceptions as return values."),

        ("event_loop_closed_before_cleanup",
         "Why does closing an asyncio event loop while tasks are pending raise `RuntimeError: Event loop is closed`?",
         "Pending asynchronous tasks or callbacks attempting to execute after loop termination fail.\n\nFix: Cancel all pending tasks with `asyncio.all_tasks()`, gather them with `return_exceptions=True`, then close the loop."),

        ("thread_unsafe_asyncio_call",
         "Why does calling `loop.create_task()` from a background OS thread cause race conditions?",
         "Asyncio event loops are not thread-safe and must be modified only from the thread running the loop.\n\nFix: Schedule from worker threads using `loop.call_soon_threadsafe()` or `asyncio.run_coroutine_threadsafe()`."),

        # Family 12: Typing, Annotations & Dataclasses (56-60)
        ("dataclass_mutable_default_field",
         "Why does `@dataclass class Config: tags: list = []` raise ValueError in Python?",
         "Dataclasses explicitly forbid mutable default values to prevent shared instance state bugs.\n\nFix: Use `field(default_factory=list)`: `tags: list = field(default_factory=list)`."),

        ("isinstance_union_type_syntax",
         "Why does `isinstance(x, int | str)` fail on Python 3.9?",
         "The pipe syntax `int | str` for `types.UnionType` was introduced in Python 3.10.\n\nFix: Use a tuple of types on older versions: `isinstance(x, (int, str))` or `from typing import Union`."),

        ("optional_type_annotation_missing_none",
         "Why does `def find_user(id: int) -> User:` cause type checker warnings if it returns `None`?",
         "The return type is annotated as `User`, which forbids returning `None`.\n\nFix: Annotate as `User | None` or `Optional[User]`."),

        ("type_guard_narrowing_failure",
         "Why does `if type(val) is Animal:` fail to match instances of subclasses like `Dog`?",
         "`type(x) is Base` checks exact class identity, ignoring inheritance.\n\nFix: Use `isinstance(val, Animal)` to support polymorphic inheritance."),

        ("runtime_type_hint_circular_import",
         "How do you resolve circular import errors caused solely by type annotations?",
         "Importing a module only for type hints at module level can trigger circular dependencies.\n\nFix: Guard the import using `if typing.TYPE_CHECKING:` and use string annotations or `from __future__ import annotations`."),

        # Family 13: Memory Management & Garbage Collection (61-65)
        ("circular_reference_with_del",
         "Why do circular object references prevent immediate garbage collection upon `del`?",
         "`del` decrements reference count. If object A references B and B references A, their reference counts remain 1 even when unreachable, requiring cyclic GC passes.\n\nFix: Break cycles using `weakref` proxies or explicit cleanup methods."),

        ("memory_leak_global_cache",
         "Why does storing items in a global dictionary `CACHE[key] = data` cause out-of-memory crashes?",
         "Global objects remain in memory for the process lifetime. Without an eviction policy (like LRU or TTL), cache size grows unbounded.\n\nFix: Use `functools.lru_cache(maxsize=1000)` or a dedicated cache with bounded size."),

        ("large_file_read_into_memory",
         "Why does `lines = open('10gb.log').readlines()` crash with MemoryError?",
         "`.readlines()` loads the entire file contents into RAM at once as a list of strings.\n\nFix: Stream line-by-line using the file iterator: `for line in open('10gb.log'): process(line)`."),

        ("sys_getsizeof_nested_containers",
         "Why does `sys.getsizeof([\"very long string\" * 1000])` return an unexpectedly small byte count?",
         "`sys.getsizeof()` computes only the size of the container itself (pointers to elements), not the memory consumed by referenced objects.\n\nFix: Use a recursive memory traversal function or a heap profiling tool like `tracemalloc`."),

        ("tracemalloc_snapshot_empty",
         "Why does `tracemalloc.take_snapshot()` return empty allocations if called immediately?",
         "Tracemalloc only tracks memory allocations made after `tracemalloc.start()` is called.\n\nFix: Ensure `tracemalloc.start()` is invoked before memory-allocating code runs."),

        # Family 14: Packaging, Imports & Module Boundaries (66-70)
        ("import_shadowed_by_local_filename",
         "Why does `import math` fail with AttributeError after creating a file named `math.py`?",
         "Python searches the current script directory first on `sys.path`. A local file named `math.py` shadows the standard library module.\n\nFix: Rename the local file to `math_utils.py` and delete any cached `__pycache__/math.*.pyc` files."),

        ("relative_import_beyond_top_level",
         "Why does `from ..module import func` raise `ValueError: attempted relative import beyond top-level package`?",
         "Relative imports cannot navigate above the root package recognized by Python's `__name__` module hierarchy.\n\nFix: Run scripts with `-m package.subpackage.module` from the repository root."),

        ("init_py_missing_submodule_export",
         "Why does `import mypkg; mypkg.submodule` fail with AttributeError if `__init__.py` is empty?",
         "Importing a parent package does not automatically import nested submodules unless they are imported inside `mypkg/__init__.py`.\n\nFix: Add `from . import submodule` inside `__init__.py`."),

        ("sys_path_append_order_hazard",
         "Why is `sys.path.append(custom_dir)` risky compared to `sys.path.insert(0, custom_dir)`?",
         "`.append()` places the directory at the end of the search path; if a module with the same name exists in site-packages, the system version is loaded first.\n\nFix: Use `sys.path.insert(0, custom_dir)` to give the custom directory precedence."),

        ("dunder_all_missing_function",
         "Why does `from mymodule import *` omit `helper()` if `__all__ = ['main']` is defined?",
         "`__all__` explicitly defines the public export symbols for wildcard imports; any symbol not listed is excluded.\n\nFix: Add `'helper'` to `__all__`: `__all__ = ['main', 'helper']`."),

        # Family 15: Functional Tools & Comprehensions (71-75)
        ("list_comprehension_variable_leak_python2_vs_3",
         "Why does `[x for x in range(5)]` not overwrite `x` in Python 3, but does in Python 2?",
         "Python 3 gives list comprehensions their own private function scope, preventing loop variables from leaking into the enclosing namespace."),

        ("map_filter_lazy_evaluation_trap",
         "Why does `res = map(func, items); len(res)` raise TypeError in Python 3?",
         "In Python 3, `map()` and `filter()` return lazy iterators rather than lists; iterators do not implement `__len__`.\n\nFix: Materialize into a list: `list(map(func, items))` or use a list comprehension."),

        ("zip_truncating_longest_iterable",
         "Why does `list(zip([1, 2, 3], ['a']))` discard `2` and `3`?",
         "Standard `zip()` truncates to the shortest input iterable by default.\n\nFix: Use `itertools.zip_longest([1, 2, 3], ['a'], fillvalue=None)` or `zip(..., strict=True)` to raise ValueError on mismatch."),

        ("reduce_initial_value_empty_sequence",
         "Why does `functools.reduce(operator.add, [])` raise a TypeError?",
         "`reduce()` requires at least one element or an explicit initial value parameter.\n\nFix: Provide an explicit initializer: `functools.reduce(operator.add, [], 0)`."),

        ("sorted_key_function_mutating_state",
         "Why does passing a stateful function as `sorted(items, key=fn)` produce non-deterministic results?",
         "Timsort invokes the key function multiple times during element comparisons; if the key function mutates internal state, sorting order becomes unstable.\n\nFix: Ensure sorting key functions are pure and stateless."),

        # Family 16: Mathematical & Algorithmic Code Pitfalls (76-80)
        ("integer_division_operator_python2_vs_3",
         "Why does `5 / 2` equal `2.5` in Python 3 while `5 // 2` equals `2`?",
         "`/` performs true floating-point division in Python 3, whereas `//` performs floor (integer) division."),

        ("bitwise_vs_logical_precedence",
         "Why does `if a & b == c:` evaluate unexpectedly in Python?",
         "Comparison operators (`==`) have higher operator precedence than bitwise operators (`&`). `a & b == c` is parsed as `a & (b == c)`.\n\nFix: Add parentheses: `(a & b) == c`."),

        ("power_operator_precedence_negative_base",
         "Why does `-2 ** 2` evaluate to `-4` instead of `4`?",
         "Exponentiation `**` has higher precedence than unary negation `-`. `-2 ** 2` evaluates as `-(2 ** 2) = -4`.\n\nFix: Use parentheses around the base: `(-2) ** 2`."),

        ("round_bankers_rounding_half_to_even",
         "Why does `round(2.5)` return 2 while `round(3.5)` returns 4 in Python?",
         "Python uses round-half-to-even (banker's rounding) to minimize statistical bias when rounding distributions.\n\nFix: If round-half-up is strictly required, use `decimal.Decimal` with `ROUND_HALF_UP`."),

        ("modulo_with_negative_numbers",
         "Why does `-7 % 3` equal `2` in Python instead of `-1`?",
         "Python defines modulo such that the result shares the sign of the divisor: `r = a - (b * (a // b))`. Since `-7 // 3 = -3`, `-7 - (3 * -3) = 2`.\n\nFix: If truncating toward zero is desired, use `math.fmod(-7, 3)`."),

        # Family 17: String Encoding & Unicode (81-85)
        ("unicode_encode_error_cp1252",
         "Why does printing Hindi Devanagari characters on Windows console crash with UnicodeEncodeError?",
         "Legacy Windows console streams default to ANSI codepages (like `cp1252`), which cannot represent characters outside the Windows Latin charset.\n\nFix: Reconfigure stdout encoding: `sys.stdout.reconfigure(encoding='utf-8')`."),

        ("utf8_bom_parsing_failure",
         "Why does reading a UTF-8 file with a Byte Order Mark (BOM) produce `\\ufeff` on the first line?",
         "Opening with `'utf-8'` leaves the 3-byte BOM sequence as a zero-width non-breaking space character.\n\nFix: Open using the `'utf-8-sig'` encoding to automatically strip BOM."),

        ("string_length_combining_characters",
         "Why does `len('e\\u0301')` return 2 for a single visible accented character 'é'?",
         "Python `len()` counts Unicode code points, not rendered grapheme clusters; the base letter `e` and combining acute accent `\\u0301` are two code points.\n\nFix: Use `unicodedata.normalize('NFC', s)` to combine into a single canonical code point."),

        ("casefold_vs_lower_unicode",
         "Why does `'ß'.lower()` equal `'ß'` while `'ß'.casefold()` equals `'ss'`?",
         "`.lower()` performs simple lowercase mapping, while `.casefold()` implements aggressive Unicode caseless matching for international text.\n\nFix: Use `.casefold()` for case-insensitive string matching across all languages."),

        ("bytearray_mutation_out_of_bounds",
         "Why does `ba = bytearray(b'abc'); ba[0] = 256` raise ValueError?",
         "A `bytearray` element must be an integer in the 8-bit range `0 <= x < 256`.\n\nFix: Mask values to 8-bit integers: `ba[0] = val & 0xFF`."),

        # Family 18: Testing & Mocking (86-90)
        ("unittest_mock_patch_wrong_target",
         "Why does `@patch('os.path.exists')` fail to mock `exists` when code does `from os.path import exists`?",
         "Mocks must be patched in the module namespace where the symbol is imported and looked up, not in the module where it was defined.\n\nFix: Patch the target module's import: `@patch('my_module.exists')`."),

        ("pytest_fixture_mutable_leak_across_tests",
         "Why does modifying a fixture object in one test corrupt data in subsequent tests?",
         "If a fixture has `scope='session'` or `scope='module'`, all tests receive the same shared instance in memory.\n\nFix: Set fixture scope to function (default): `@pytest.fixture(scope='function')` or return a factory function."),

        ("mock_assert_called_typo_silent_pass",
         "Why does `mock.assert_called_once` silently pass without verifying anything?",
         "`assert_called_once` is a method and must be called with parentheses: `mock.assert_called_once()`. Referencing it as an attribute evaluates to a truthy function object.\n\nFix: Call the method with parentheses, or use `pytest-mock`."),

        ("test_assertion_order_equality",
         "Why does `assert [1, 2] == [2, 1]` fail if order does not matter in the test?",
         "List equality checks strict element ordering.\n\nFix: Compare as sets `set(a) == set(b)` or use `sorted(a) == sorted(b)` if duplicates exist."),

        ("floating_point_assertion_pytest_approx",
         "Why does `assert 0.1 + 0.2 == 0.3` fail inside a pytest test?",
         "Direct floating point comparison fails due to binary representation limits.\n\nFix: Use `pytest.approx`: `assert 0.1 + 0.2 == pytest.approx(0.3)`."),

        # Family 19: Object Identity & Comparison (91-95)
        ("nan_equality_reflexivity",
         "Why does `float('nan') == float('nan')` return False in Python?",
         "The IEEE 754 standard defines NaN (Not-a-Number) as non-reflexive; NaN is not equal to any value, including itself.\n\nFix: Test for NaN using `math.isnan(x)`."),

        ("chained_comparison_short_circuit",
         "Why does `1 < x < 5` behave differently from `(1 < x) and (x < 5)` when x is a function call?",
         "Python chained comparisons evaluate middle subexpressions only once: `1 < f() < 5` invokes `f()` exactly once, whereas separate comparisons would call `f()` twice."),

        ("boolean_is_subclass_of_int",
         "Why does `isinstance(True, int)` return True in Python?",
         "`bool` is an explicit subclass of `int` in Python (`True == 1`, `False == 0`).\n\nFix: If distinguishing booleans from integers is required, check `type(x) is int`."),

        ("singleton_comparison_none_identity",
         "Why is `x == None` considered bad style compared to `x is None`?",
         "Objects can override `__eq__` to return True when compared with None, whereas `is None` directly checks identity against the unique `NoneType` singleton.\n\nFix: Always write `if x is None:`."),

        ("custom_class_default_identity_comparison",
         "Why does `Point(1, 2) == Point(1, 2)` return False for a basic class?",
         "Classes that do not implement `__eq__` inherit object identity comparison from `object`, returning True only if both references point to the same memory instance.\n\nFix: Implement `__eq__` or use a `@dataclass`."),

        # Family 20: System, OS & Process (96-100)
        ("os_environ_mutation_child_vs_parent",
         "Why does setting `os.environ['KEY'] = 'val'` inside a Python script not persist in the calling shell?",
         "Environment variables are inherited by child processes upon creation; a child process cannot alter the environment of its parent process.\n\nFix: Shell environments must be configured in the calling shell or sourced via scripts."),

        ("file_path_separator_cross_platform",
         "Why does `'dir/' + filename` break on Windows when passed to certain OS APIs?",
         "Hardcoding forward slashes or backslashes causes portability issues across Windows and POSIX systems.\n\nFix: Use `pathlib.Path`: `Path('dir') / filename`."),

        ("signal_handling_non_main_thread",
         "Why does `signal.signal(signal.SIGINT, handler)` raise ValueError inside a worker thread?",
         "In Python, signal handlers can only be registered and executed within the main thread of the process.\n\nFix: Register signal handlers in the main thread before starting background worker threads."),

        ("tempfile_deletion_before_close_windows",
         "Why does opening a `NamedTemporaryFile` by path raise PermissionError on Windows?",
         "On Windows, a file cannot be opened by a second process or handle while held open by `NamedTemporaryFile` without delete sharing.\n\nFix: Use `delete=False`, explicitly close the file handle, and delete manually using `os.unlink()` in a finally block."),

        ("shutil_rmtree_symlink_safety",
         "Why can `shutil.rmtree(path)` delete target directory contents if path is a directory symlink?",
         "In some configurations and platforms, removing a symlinked tree can traverse into the real directory target.\n\nFix: Check if the path is a symlink first: `if path.is_symlink(): path.unlink() else: shutil.rmtree(path)`.")
    ]

    assert len(bugs) == 100, f"Expected 100 bug diagnostics, got {len(bugs)}"

    records = []
    for i, (title, prompt, response) in enumerate(bugs, 1):
        records.append({
            "id": f"EXP34-DBUG-{i:04d}",
            "prompt": prompt,
            "response": response,
            "category": "code_debugging",
            "source_type": "synthetic_code_curation",
            "language": "en",
            "license": "CC0-1.0",
            "data_role": "TRAIN",
            "review_status": "VERIFIED"
        })
    return records

