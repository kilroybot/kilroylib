import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from types import TracebackType
from typing import (
    Any,
    AsyncContextManager,
    Awaitable,
    ContextManager,
    IO,
    Optional,
    Type,
    TypeVar,
    Union,
)

import aiofiles
import dill
from aiofiles.base import AiofilesContextManager
from aiofiles.threadpool.binary import AsyncBufferedIOBase, AsyncFileIO

T = TypeVar("T")
C = TypeVar("C", bound="Contextable")


class anullcontext(nullcontext):
    async def __aenter__(self):
        return self.enter_result

    async def __aexit__(self, *excinfo):
        pass


class Contextable:
    async def __aenter__(self: C) -> C:
        return self

    async def __aexit__(
        self,
        exctype: Optional[Type[BaseException]],
        excinst: Optional[BaseException],
        exctb: Optional[TracebackType],
    ) -> bool:
        return False  # don't suppress exceptions

    def __enter__(self: C) -> C:
        return run_sync(self.__aenter__())

    def __exit__(
        self,
        exctype: Optional[Type[BaseException]],
        excinst: Optional[BaseException],
        exctb: Optional[TracebackType],
    ) -> bool:
        return run_sync(self.__aexit__(exctype, excinst, exctb))


def file_context(
    file: Union[AsyncFileIO, str, Path],
    mode: Optional[str] = None,
) -> Union[
    AsyncContextManager[AsyncFileIO],
    AiofilesContextManager[None, None, AsyncBufferedIOBase],
]:
    """
    Gets convenient context manager for multi-type file argument.

    If the file is a path then context manager opens and closes the file.
    If the file is already open file handle then context manager does nothing.

    Args:
        file (Union[IO, str, Path]): Either a path to the file
            or an open file handle.
        mode (Optional[str]): Mode to open the file in.
            Only used when file is a path.

    Returns:
        ContextManager[IO]: Context Manager that ensures the file is open.
    """
    if isinstance(file, (str, Path)):
        return aiofiles.open(file, mode)
    return anullcontext(file)


async def safe_dump(obj: Any, file: Union[AsyncFileIO, str, Path]) -> None:
    """Dumps object to file using dill.

    Args:
        obj (Any): Any object that can be dumped with dill.
        file (Union[AsyncFileIO, str, Path]): File to dump to.
            Either a path to the file or an open file handle.
    """
    async with file_context(file, "wb") as f:
        content = dill.dumps(obj, protocol=dill.HIGHEST_PROTOCOL, recurse=True)
        await f.write(content)


async def safe_load(file: Union[AsyncFileIO, str, Path]) -> Any:
    """Loads object from file using dill.

    Args:
        file (Union[AsyncFileIO, str, Path]): File to load from.
            Either a path to the file or an open file handle.

    Returns:
        Any: Object loaded from the file.
    """
    async with file_context(file, "rb") as f:
        return dill.loads(await f.read())


def run_sync(awaitable: Awaitable[T]) -> T:
    def run():
        return asyncio.new_event_loop().run_until_complete(awaitable)

    with ThreadPoolExecutor() as executor:
        future = executor.submit(run)
        return future.result()
