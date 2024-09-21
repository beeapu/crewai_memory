# crewai_memory
# What they are storing and how they are storing?

-   They have imported :mem0ai : Long-term memory for AI Agents ( planning to add in new version) 
-   2 places memory is accessed
	-   Launching the crew:
	-   Creating the agent:
- MEMORY
	-
	- memory class:
	``` python
	from typing import Any, Dict, Optional
	from crewai.memory.storage.interface import Storage
	class Memory:
    """
    Base class for memory, now supporting agent tags and generic metadata.
    """
    def __init__(self, storage: Storage):
        self.storage = storage

    def save(
        self,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
        agent: Optional[str] = None,
    ) -> None:
        metadata = metadata or {}
        if agent:
            metadata["agent"] = agent

        self.storage.save(value, metadata)

    def search(self, query: str) -> Dict[str, Any]:
        return self.storage.search(query)
   ```
- Short Term memory: -> Rag storage
```python
from typing import Any, Dict, Optional
class ShortTermMemoryItem:
    def __init__(
        self,
        data: Any,
        agent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.data = data
        self.agent = agent
        self.metadata = metadata if metadata is not None else {}

```
``` python 
from typing import Any, Dict, Optional
from crewai.memory.memory import Memory
from crewai.memory.short_term.short_term_memory_item import ShortTermMemoryItem
from crewai.memory.storage.rag_storage import RAGStorage

class ShortTermMemory(Memory):
    """
    ShortTermMemory class for managing transient data related to immediate tasks
    and interactions.
    Inherits from the Memory class and utilizes an instance of a class that
    adheres to the Storage for data storage, specifically working with
    MemoryItem instances.
    """

    def __init__(self, crew=None, embedder_config=None):
        storage = RAGStorage(
            type="short_term", embedder_config=embedder_config, crew=crew
        )
        super().__init__(storage)

    def save(
        self,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
        agent: Optional[str] = None,
    ) -> None:
        item = ShortTermMemoryItem(data=value, metadata=metadata, agent=agent)

        super().save(value=item.data, metadata=item.metadata, agent=item.agent)

    def search(self, query: str, score_threshold: float = 0.35):
        return self.storage.search(query=query, score_threshold=score_threshold)  # type: ignore # BUG? The reference is to the parent class, but the parent class does not have this parameters

    def reset(self) -> None:
        try:
            self.storage.reset()
        except Exception as e:
            raise Exception(
                f"An error occurred while resetting the short-term memory: {e}"
            )
```
  - Long term memory: -> SQL storage
```python
from typing import Any, Dict
from crewai.memory.long_term.long_term_memory_item import LongTermMemoryItem
from crewai.memory.memory import Memory
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
class LongTermMemory(Memory):
    """
    LongTermMemory class for managing cross runs data related to overall crew's
    execution and performance.
    Inherits from the Memory class and utilizes an instance of a class that
    adheres to the Storage for data storage, specifically working with
    LongTermMemoryItem instances.
    """

    def __init__(self):
        storage = LTMSQLiteStorage()
        super().__init__(storage)

    def save(self, item: LongTermMemoryItem) -> None:  # type: ignore # BUG?: Signature of "save" incompatible with supertype "Memory"
        metadata = item.metadata
        metadata.update({"agent": item.agent, "expected_output": item.expected_output})
        self.storage.save(  # type: ignore # BUG?: Unexpected keyword argument "task_description","score","datetime" for "save" of "Storage"
            task_description=item.task,
            score=metadata["quality"],
            metadata=metadata,
            datetime=item.datetime,
        )
    def search(self, task: str, latest_n: int = 3) -> Dict[str, Any]:
        return self.storage.load(task, latest_n)  # type: ignore # BUG?: "Storage" has no attribute "load"

    def reset(self) -> None:
        self.storage.reset()
```
```python
# long term item 
from typing import Any, Dict, Optional, Union
class LongTermMemoryItem:
    def __init__(
        self,
        agent: str,
        task: str,
        expected_output: str,
        datetime: str,
        quality: Optional[Union[int, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.task = task
        self.agent = agent
        self.quality = quality
        self.datetime = datetime
        self.expected_output = expected_output
        self.metadata = metadata if metadata is not None else {}

```
File to store the memory in Lont term with sq light: 
```python
import json
import sqlite3
from typing import Any, Dict, List, Optional, Union

from crewai.utilities import Printer
from crewai.utilities.paths import db_storage_path


class LTMSQLiteStorage:
    """
    An updated SQLite storage class for LTM data storage.
    """

    def __init__(
        self, db_path: str = f"{db_storage_path()}/long_term_memory_storage.db"
    ) -> None:
        self.db_path = db_path
        self._printer: Printer = Printer()
        self._initialize_db()

    def _initialize_db(self):
        """
        Initializes the SQLite database and creates LTM table
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS long_term_memories (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        task_description TEXT,
                        metadata TEXT,
                        datetime TEXT,
                        score REAL
                    )
                """
                )

                conn.commit()
        except sqlite3.Error as e:
            self._printer.print(
                content=f"MEMORY ERROR: An error occurred during database initialization: {e}",
                color="red",
            )

    def save(
        self,
        task_description: str,
        metadata: Dict[str, Any],
        datetime: str,
        score: Union[int, float],
    ) -> None:
        """Saves data to the LTM table with error handling."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                INSERT INTO long_term_memories (task_description, metadata, datetime, score)
                VALUES (?, ?, ?, ?)
            """,
                    (task_description, json.dumps(metadata), datetime, score),
                )
                conn.commit()
        except sqlite3.Error as e:
            self._printer.print(
                content=f"MEMORY ERROR: An error occurred while saving to LTM: {e}",
                color="red",
            )

    def load(
        self, task_description: str, latest_n: int
    ) -> Optional[List[Dict[str, Any]]]:
        """Queries the LTM table by task description with error handling."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"""
                    SELECT metadata, datetime, score
                    FROM long_term_memories
                    WHERE task_description = ?
                    ORDER BY datetime DESC, score ASC
                    LIMIT {latest_n}
                """,
                    (task_description,),
                )
                rows = cursor.fetchall()
                if rows:
                    return [
                        {
                            "metadata": json.loads(row[0]),
                            "datetime": row[1],
                            "score": row[2],
                        }
                        for row in rows
                    ]

        except sqlite3.Error as e:
            self._printer.print(
                content=f"MEMORY ERROR: An error occurred while querying LTM: {e}",
                color="red",
            )
        return None

    def reset(
        self,
    ) -> None:
        """Resets the LTM table with error handling."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM long_term_memories")
                conn.commit()

        except sqlite3.Error as e:
            self._printer.print(
                content=f"MEMORY ERROR: An error occurred while deleting all rows in LTM: {e}",
                color="red",
            )
        return None

```
 - Entity Memory: -> Rag storage
``` python
from crewai.memory.entity.entity_memory_item import EntityMemoryItem
from crewai.memory.memory import Memory
from crewai.memory.storage.rag_storage import RAGStorage
class EntityMemory(Memory):
    """
    EntityMemory class for managing structured information about entities
    and their relationships using SQLite storage.
    Inherits from the Memory class.
    """

    def __init__(self, crew=None, embedder_config=None):
        storage = RAGStorage(
            type="entities",
            allow_reset=False,
            embedder_config=embedder_config,
            crew=crew,
        )
        super().__init__(storage)

    def save(self, item: EntityMemoryItem) -> None:  # type: ignore # BUG?: Signature of "save" incompatible with supertype "Memory"
        """Saves an entity item into the SQLite storage."""
        data = f"{item.name}({item.type}): {item.description}"
        super().save(data, item.metadata)

    def reset(self) -> None:
        try:
            self.storage.reset()
        except Exception as e:
            raise Exception(f"An error occurred while resetting the entity memory: {e}")
```
``` python
class EntityMemoryItem:
    def __init__(
        self,
        name: str,
        type: str,
        description: str,
        relationships: str,
    ):
        self.name = name
        self.type = type
        self.description = description
        self.metadata = {"relationships": relationships}
```
- context memory:
```python
from typing import Optional
from crewai.memory import EntityMemory, LongTermMemory, ShortTermMemory
class ContextualMemory:
    def __init__(self, stm: ShortTermMemory, ltm: LongTermMemory, em: EntityMemory):
        self.stm = stm
        self.ltm = ltm
        self.em = em

    def build_context_for_task(self, task, context) -> str:
        """
        Automatically builds a minimal, highly relevant set of contextual information
        for a given task.
        """
        query = f"{task.description} {context}".strip()
        if query == "":
            return ""
        context = []
        context.append(self._fetch_ltm_context(task.description))
        context.append(self._fetch_stm_context(query))
        context.append(self._fetch_entity_context(query))
        return "\n".join(filter(None, context))
    def _fetch_stm_context(self, query) -> str:
        """
        Fetches recent relevant insights from STM related to the task's description and expected_output,
        formatted as bullet points.
        """
        stm_results = self.stm.search(query)
        formatted_results = "\n".join([f"- {result}" for result in stm_results])
        return f"Recent Insights:\n{formatted_results}" if stm_results else ""
    def _fetch_ltm_context(self, task) -> Optional[str]:
        """
        Fetches historical data or insights from LTM that are relevant to the task's description and expected_output,
        formatted as bullet points.
        """
        ltm_results = self.ltm.search(task, latest_n=2)
        if not ltm_results:
            return None

        formatted_results = [
            suggestion
            for result in ltm_results
            for suggestion in result["metadata"]["suggestions"]  # type: ignore # Invalid index type "str" for "str"; expected type "SupportsIndex | slice"
        ]
        formatted_results = list(dict.fromkeys(formatted_results))
        formatted_results = "\n".join([f"- {result}" for result in formatted_results])  # type: ignore # Incompatible types in assignment (expression has type "str", variable has type "list[str]")

        return f"Historical Data:\n{formatted_results}" if ltm_results else ""

    def _fetch_entity_context(self, query) -> str:
        """
        Fetches relevant entity information from Entity Memory related to the task's description and expected_output,
        formatted as bullet points.
        """
        em_results = self.em.search(query)
        formatted_results = "\n".join(
            [f"- {result['context']}" for result in em_results]  # type: ignore #  Invalid index type "str" for "str"; expected type "SupportsIndex | slice"
        )
        return f"Entities:\n{formatted_results}" if em_results else ""
```

RAG STORAGE CODE USING EMBIDDING CHAIN:
``` python
import contextlib
import io
import logging
import os
import shutil
from typing import Any, Dict, List, Optional

from embedchain import App
from embedchain.llm.base import BaseLlm
from embedchain.models.data_type import DataType
from embedchain.vectordb.chroma import InvalidDimensionException

from crewai.memory.storage.interface import Storage
from crewai.utilities.paths import db_storage_path


@contextlib.contextmanager
def suppress_logging(
    logger_name="chromadb.segment.impl.vector.local_persistent_hnsw",
    level=logging.ERROR,
):
    logger = logging.getLogger(logger_name)
    original_level = logger.getEffectiveLevel()
    logger.setLevel(level)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ), contextlib.suppress(UserWarning):
        yield
    logger.setLevel(original_level)


class FakeLLM(BaseLlm):
    pass


class RAGStorage(Storage):
    """
    Extends Storage to handle embeddings for memory entries, improving
    search efficiency.
    """

    def __init__(self, type, allow_reset=True, embedder_config=None, crew=None):
        super().__init__()
        if (
            not os.getenv("OPENAI_API_KEY")
            and not os.getenv("OPENAI_BASE_URL") == "https://api.openai.com/v1"
        ):
            os.environ["OPENAI_API_KEY"] = "fake"

        agents = crew.agents if crew else []
        agents = [self._sanitize_role(agent.role) for agent in agents]
        agents = "_".join(agents)

        config = {
            "app": {
                "config": {"name": type, "collect_metrics": False, "log_level": "ERROR"}
            },
            "chunker": {
                "chunk_size": 5000,
                "chunk_overlap": 100,
                "length_function": "len",
                "min_chunk_size": 150,
            },
            "vectordb": {
                "provider": "chroma",
                "config": {
                    "collection_name": type,
                    "dir": f"{db_storage_path()}/{type}/{agents}",
                    "allow_reset": allow_reset,
                },
            },
        }

        if embedder_config:
            config["embedder"] = embedder_config
        self.type = type
        self.app = App.from_config(config=config)
        self.app.llm = FakeLLM()
        if allow_reset:
            self.app.reset()

    def _sanitize_role(self, role: str) -> str:
        """
        Sanitizes agent roles to ensure valid directory names.
        """
        return role.replace("\n", "").replace(" ", "_").replace("/", "_")

    def save(self, value: Any, metadata: Dict[str, Any]) -> None:
        self._generate_embedding(value, metadata)

    def search(  # type: ignore # BUG?: Signature of "search" incompatible with supertype "Storage"
        self,
        query: str,
        limit: int = 3,
        filter: Optional[dict] = None,
        score_threshold: float = 0.35,
    ) -> List[Any]:
        with suppress_logging():
            try:
                results = (
                    self.app.search(query, limit, where=filter)
                    if filter
                    else self.app.search(query, limit)
                )
            except InvalidDimensionException:
                self.app.reset()
                return []
        return [r for r in results if r["metadata"]["score"] >= score_threshold]

    def _generate_embedding(self, text: str, metadata: Dict[str, Any]) -> Any:
        self.app.add(text, data_type=DataType.TEXT, metadata=metadata)

    def reset(self) -> None:
        try:
            shutil.rmtree(f"{db_storage_path()}/{self.type}")
        except Exception as e:
            raise Exception(
                f"An error occurred while resetting the {self.type} memory: {e}"
            )
```
Long term memory storage code using sqlit
``` python
import json
import sqlite3
from typing import Any, Dict, List, Optional, Union

from crewai.utilities import Printer
from crewai.utilities.paths import db_storage_path


class LTMSQLiteStorage:
    """
    An updated SQLite storage class for LTM data storage.
    """

    def __init__(
        self, db_path: str = f"{db_storage_path()}/long_term_memory_storage.db"
    ) -> None:
        self.db_path = db_path
        self._printer: Printer = Printer()
        self._initialize_db()

    def _initialize_db(self):
        """
        Initializes the SQLite database and creates LTM table
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS long_term_memories (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        task_description TEXT,
                        metadata TEXT,
                        datetime TEXT,
                        score REAL
                    )
                """
                )

                conn.commit()
        except sqlite3.Error as e:
            self._printer.print(
                content=f"MEMORY ERROR: An error occurred during database initialization: {e}",
                color="red",
            )

    def save(
        self,
        task_description: str,
        metadata: Dict[str, Any],
        datetime: str,
        score: Union[int, float],
    ) -> None:
        """Saves data to the LTM table with error handling."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                INSERT INTO long_term_memories (task_description, metadata, datetime, score)
                VALUES (?, ?, ?, ?)
            """,
                    (task_description, json.dumps(metadata), datetime, score),
                )
                conn.commit()
        except sqlite3.Error as e:
            self._printer.print(
                content=f"MEMORY ERROR: An error occurred while saving to LTM: {e}",
                color="red",
            )

    def load(
        self, task_description: str, latest_n: int
    ) -> Optional[List[Dict[str, Any]]]:
        """Queries the LTM table by task description with error handling."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"""
                    SELECT metadata, datetime, score
                    FROM long_term_memories
                    WHERE task_description = ?
                    ORDER BY datetime DESC, score ASC
                    LIMIT {latest_n}
                """,
                    (task_description,),
                )
                rows = cursor.fetchall()
                if rows:
                    return [
                        {
                            "metadata": json.loads(row[0]),
                            "datetime": row[1],
                            "score": row[2],
                        }
                        for row in rows
                    ]

        except sqlite3.Error as e:
            self._printer.print(
                content=f"MEMORY ERROR: An error occurred while querying LTM: {e}",
                color="red",
            )
        return None

    def reset(
        self,
    ) -> None:
        """Resets the LTM table with error handling."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM long_term_memories")
                conn.commit()

        except sqlite3.Error as e:
            self._printer.print(
                content=f"MEMORY ERROR: An error occurred while deleting all rows in LTM: {e}",
                color="red",
            )
        return None
```
