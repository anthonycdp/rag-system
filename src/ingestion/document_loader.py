"""Document loader for various file formats."""

from pathlib import Path
from typing import Iterator

from langchain_community.document_loaders import (
    CSVLoader,
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document
from pydantic import BaseModel


class DocumentLoaderConfig(BaseModel):
    """Configuration for document loading."""

    file_patterns: dict[str, str] = {
        "*.pdf": "pdf",
        "*.txt": "text",
        "*.md": "markdown",
        "*.csv": "csv",
    }
    encoding: str = "utf-8"
    show_progress: bool = True


class DocumentLoader:
    """Load documents from various sources and formats."""

    def __init__(self, config: DocumentLoaderConfig | None = None) -> None:
        """Initialize the document loader.

        Args:
            config: Configuration for document loading.
        """
        self.config = config or DocumentLoaderConfig()

    def load_file(self, file_path: Path) -> list[Document]:
        """Load a single file.

        Args:
            file_path: Path to the file to load.

        Returns:
            List of Document objects.

        Raises:
            ValueError: If file type is not supported.
        """
        suffix = file_path.suffix.lower()
        loader_map: dict[str, type] = {
            ".pdf": PyPDFLoader,
            ".txt": TextLoader,
            ".md": UnstructuredMarkdownLoader,
            ".csv": CSVLoader,
        }

        if suffix not in loader_map:
            raise ValueError(f"Unsupported file type: {suffix}")

        loader_class = loader_map[suffix]

        if suffix == ".txt":
            loader = loader_class(str(file_path), encoding=self.config.encoding)
        else:
            loader = loader_class(str(file_path))

        return loader.load()

    def load_directory(
        self, directory: Path, recursive: bool = True
    ) -> list[Document]:
        """Load all supported documents from a directory.

        Args:
            directory: Path to the directory.
            recursive: Whether to search recursively.

        Returns:
            List of Document objects from all files.
        """
        all_documents: list[Document] = []

        for pattern, _ in self.config.file_patterns.items():
            glob_pattern = f"**/{pattern}" if recursive else pattern
            loader = DirectoryLoader(
                str(directory),
                glob=glob_pattern,
                show_progress=self.config.show_progress,
            )
            documents = loader.load()
            all_documents.extend(documents)

        return all_documents

    def load_from_source(
        self, source: Path | str, recursive: bool = True
    ) -> list[Document]:
        """Load documents from a file or directory.

        Args:
            source: Path to file or directory.
            recursive: Whether to search directories recursively.

        Returns:
            List of Document objects.
        """
        source_path = Path(source)

        if source_path.is_file():
            return self.load_file(source_path)
        elif source_path.is_dir():
            return self.load_directory(source_path, recursive)
        else:
            raise FileNotFoundError(f"Source not found: {source}")

    def lazy_load(self, source: Path) -> Iterator[Document]:
        """Lazily load documents one at a time.

        Useful for large document collections.

        Args:
            source: Path to file or directory.

        Yields:
            Document objects one at a time.
        """
        if source.is_file():
            yield from self.load_file(source)
        elif source.is_dir():
            for pattern in self.config.file_patterns:
                for file_path in source.glob(f"**/{pattern}" if source else pattern):
                    try:
                        yield from self.load_file(file_path)
                    except Exception:
                        continue


def load_sample_documents() -> list[Document]:
    """Create sample documents for demonstration.

    Returns:
        List of sample Document objects about Python programming.
    """
    sample_texts = [
        {
            "content": """
Python Basics: Variables and Data Types

Python is a dynamically typed programming language, meaning you don't need to declare
variable types explicitly. Python supports several built-in data types:

1. Numeric Types:
   - int: Integer numbers (e.g., 42, -17)
   - float: Floating-point numbers (e.g., 3.14, -0.001)
   - complex: Complex numbers (e.g., 3+4j)

2. Sequence Types:
   - str: Strings, immutable sequences of characters
   - list: Mutable ordered collections
   - tuple: Immutable ordered collections

3. Mapping Types:
   - dict: Key-value pairs, mutable

4. Set Types:
   - set: Unordered collection of unique elements
   - frozenset: Immutable version of set

5. Boolean Type:
   - bool: True or False values

Variables are created simply by assigning a value:
    x = 42
    name = "Python"
    prices = [19.99, 29.99, 39.99]
""",
            "metadata": {"source": "python_basics.txt", "topic": "variables"},
        },
        {
            "content": """
Python Functions and Control Flow

Functions in Python are defined using the 'def' keyword. They allow you to encapsulate
reusable blocks of code.

Basic Function Syntax:
    def function_name(parameters):
        \"\"\"Docstring describing the function.\"\"\"
        # Function body
        return result

Control Flow Statements:

1. Conditional Statements (if/elif/else):
    if condition:
        # code block
    elif another_condition:
        # code block
    else:
        # code block

2. For Loops:
    for item in iterable:
        # process item

3. While Loops:
    while condition:
        # code block

4. List Comprehensions:
    squares = [x**2 for x in range(10)]
    evens = [x for x in range(20) if x % 2 == 0]

5. Exception Handling:
    try:
        # code that might raise an exception
    except ValueError as e:
        # handle specific exception
    finally:
        # cleanup code
""",
            "metadata": {"source": "python_functions.txt", "topic": "functions"},
        },
        {
            "content": """
Object-Oriented Programming in Python

Python supports object-oriented programming (OOP) with classes and objects.

Class Definition:
    class ClassName:
        \"\"\"Class docstring.\"\"\"

        def __init__(self, attribute1, attribute2):
            self.attribute1 = attribute1
            self.attribute2 = attribute2

        def method(self):
            return self.attribute1

Key OOP Concepts in Python:

1. Inheritance:
    class ChildClass(ParentClass):
        pass

2. Multiple Inheritance:
    class ChildClass(Parent1, Parent2):
        pass

3. Encapsulation:
   - Use underscore prefix for "protected" members: _protected
   - Use double underscore for "private" members: __private

4. Polymorphism:
   - Method overriding in child classes
   - Duck typing: "If it walks like a duck..."

5. Special Methods (Dunder Methods):
   - __str__: String representation
   - __repr__: Official representation
   - __len__: For len() function
   - __getitem__: For indexing
   - __iter__: For iteration

6. Property Decorators:
    @property
    def attribute(self):
        return self._attribute

    @attribute.setter
    def attribute(self, value):
        self._attribute = value
""",
            "metadata": {"source": "python_oop.txt", "topic": "oop"},
        },
        {
            "content": """
Python Data Structures and Algorithms

Built-in Data Structures:

1. Lists: Dynamic arrays
   - append(), extend(), insert()
   - remove(), pop(), clear()
   - index(), count(), sort(), reverse()
   - Time complexity: O(1) for append, O(n) for insert/remove

2. Dictionaries: Hash maps
   - keys(), values(), items()
   - get(), update(), pop()
   - Time complexity: O(1) average for lookup/insert/delete

3. Sets: Unordered unique collections
   - add(), remove(), discard()
   - union(), intersection(), difference()
   - Time complexity: O(1) average for membership test

4. Tuples: Immutable sequences
   - Cannot be modified after creation
   - Can be used as dictionary keys
   - Time complexity: O(1) for indexing

Advanced Data Structures (from collections module):

- deque: Double-ended queue for fast append/pop from both ends
- defaultdict: Dictionary with default values
- Counter: Count hashable objects
- OrderedDict: Dictionary that maintains insertion order
- namedtuple: Tuple with named fields

Common Algorithm Patterns:

1. Two Pointers: For array/string problems
2. Sliding Window: For subarray/substring problems
3. Binary Search: O(log n) search in sorted arrays
4. DFS/BFS: Graph traversal
5. Dynamic Programming: Optimal substructure problems
""",
            "metadata": {"source": "python_data_structures.txt", "topic": "data_structures"},
        },
        {
            "content": """
Python File Handling and I/O

File Operations:

1. Reading Files:
    # Using with statement (recommended)
    with open('file.txt', 'r') as f:
        content = f.read()  # Read entire file
        lines = f.readlines()  # Read all lines
        line = f.readline()  # Read one line

2. Writing Files:
    with open('file.txt', 'w') as f:
        f.write('Hello, World!')
        f.writelines(['line1\\n', 'line2\\n'])

3. File Modes:
   - 'r': Read (default)
   - 'w': Write (truncates file)
   - 'a': Append
   - 'x': Exclusive creation
   - 'b': Binary mode
   - '+': Read and write

4. Working with Paths (pathlib):
    from pathlib import Path
    p = Path('data/file.txt')
    p.exists()
    p.is_file()
    p.read_text()
    p.write_text('content')

5. JSON Handling:
    import json
    data = json.loads(json_string)  # Parse JSON
    json_string = json.dumps(data)  # Serialize

    with open('data.json', 'r') as f:
        data = json.load(f)

6. CSV Handling:
    import csv
    with open('data.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            print(row)
""",
            "metadata": {"source": "python_file_io.txt", "topic": "file_io"},
        },
        {
            "content": """
Python Error Handling and Debugging

Exception Hierarchy:
- BaseException
  - Exception
    - ValueError, TypeError, NameError
    - IndexError, KeyError
    - IOError, OSError
    - RuntimeError

Best Practices for Error Handling:

1. Be Specific:
    # Good
    except ValueError as e:

    # Avoid
    except Exception:

2. Use Context Managers:
    with open('file.txt') as f:  # Auto-closes file
        content = f.read()

3. Custom Exceptions:
    class CustomError(Exception):
        def __init__(self, message, code):
            super().__init__(message)
            self.code = code

4. Assertions (for debugging):
    assert condition, "Error message"

5. Logging:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Message")
    logger.error("Error occurred", exc_info=True)

Debugging Tools:

1. print() debugging
2. logging module
3. pdb debugger:
    import pdb; pdb.set_trace()
4. IDE debuggers
5. Python debugger (pdb) commands:
   - n: next line
   - c: continue
   - p variable: print variable
   - l: list code
   - q: quit
""",
            "metadata": {"source": "python_error_handling.txt", "topic": "error_handling"},
        },
        {
            "content": """
Python Concurrency and Parallelism

Three Main Approaches:

1. Threading (for I/O-bound tasks):
    import threading

    def worker():
        print("Working")

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()

    # Thread Pool
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(function, items)

2. Multiprocessing (for CPU-bound tasks):
    from multiprocessing import Process, Pool

    def worker():
        print("Working")

    process = Process(target=worker)
    process.start()
    process.join()

    # Process Pool
    with Pool(processes=4) as pool:
        results = pool.map(function, items)

3. Asyncio (for async I/O):
    import asyncio

    async def fetch_data():
        await asyncio.sleep(1)
        return "data"

    async def main():
        result = await fetch_data()

    asyncio.run(main())

Key Differences:
- Threading: Good for I/O, limited by GIL for CPU
- Multiprocessing: True parallelism, separate memory
- Asyncio: Single-threaded, cooperative multitasking

When to Use:
- I/O-bound: Threading or Asyncio
- CPU-bound: Multiprocessing
- Mixed: Consider process pool + threads
""",
            "metadata": {"source": "python_concurrency.txt", "topic": "concurrency"},
        },
        {
            "content": """
Python Testing Best Practices

Testing Frameworks:

1. unittest (Built-in):
    import unittest

    class TestMath(unittest.TestCase):
        def setUp(self):
            self.value = 10

        def test_addition(self):
            self.assertEqual(1 + 1, 2)

        def tearDown(self):
            pass

    if __name__ == '__main__':
        unittest.main()

2. pytest (Recommended):
    def test_addition():
        assert 1 + 1 == 2

    def test_with_fixture(tmp_path):
        file_path = tmp_path / "test.txt"
        file_path.write_text("content")

Test Types:
- Unit Tests: Test individual functions/methods
- Integration Tests: Test component interactions
- End-to-End Tests: Test complete workflows

Test Coverage:
    pip install pytest-cov
    pytest --cov=my_package tests/

Best Practices:

1. AAA Pattern:
   - Arrange: Set up test data
   - Act: Execute the code
   - Assert: Check results

2. Test Naming:
   - test_<function>_<scenario>_<expected>

3. Use Fixtures for setup/teardown
4. Mock external dependencies
5. Test edge cases and errors
6. Keep tests independent
7. One assertion per test (when possible)
""",
            "metadata": {"source": "python_testing.txt", "topic": "testing"},
        },
    ]

    return [
        Document(page_content=text["content"], metadata=text["metadata"])
        for text in sample_texts
    ]
