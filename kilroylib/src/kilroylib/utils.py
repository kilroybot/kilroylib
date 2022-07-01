import asyncio
from asyncio import AbstractEventLoop, get_running_loop
from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from types import TracebackType
from typing import (
    Any,
    AsyncContextManager,
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    ContextManager,
    IO,
    Iterable,
    Optional,
    Tuple,
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
    ) -> Optional[bool]:
        return False  # don't suppress exceptions

    def __enter__(self: C) -> C:
        return run_sync(self.__aenter__())

    def __exit__(
        self,
        exctype: Optional[Type[BaseException]],
        excinst: Optional[BaseException],
        exctb: Optional[TracebackType],
    ) -> Optional[bool]:
        return run_sync(self.__aexit__(exctype, excinst, exctb))


def file_context(
    file: Union[AsyncFileIO, str, Path],
    mode: str = "r",
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


async def aenumerate(
    iterable: AsyncIterable[T],
) -> AsyncIterator[Tuple[int, T]]:
    i = 0
    async for x in iterable:
        yield i, x
        i += 1


def run_sync(
    awaitable: Awaitable[T], executor: Optional[Executor] = None
) -> T:
    def run():
        return asyncio.new_event_loop().run_until_complete(awaitable)

    if executor is not None:
        return executor.submit(run).result()

    with ThreadPoolExecutor() as executor:
        return executor.submit(run).result()


async def background(
    f: Callable[..., T],
    *args,
    loop: Optional[AbstractEventLoop] = None,
    executor: Optional[Executor] = None,
    **kwargs,
) -> T:
    loop = loop or get_running_loop()
    f = partial(f, *args, **kwargs)
    return await loop.run_in_executor(executor, f)


async def abackground(
    a: Awaitable[T],
    loop: Optional[AbstractEventLoop] = None,
    executor: Optional[Executor] = None,
) -> T:
    loop = loop or get_running_loop()
    return await loop.run_in_executor(executor, run_sync, a, executor)


def asyncify(
    it: Union[Iterable[T], AsyncIterable[T], Awaitable[Iterable[T]]]
) -> AsyncIterable[T]:
    if isinstance(it, AsyncIterable):
        return it

    async def wrap_sync(
        it: Union[Iterable[T], Awaitable[Iterable[T]]]
    ) -> AsyncIterable[T]:
        if isinstance(it, Awaitable):
            it = await it
        for item in it:
            yield item

    return wrap_sync(it)
