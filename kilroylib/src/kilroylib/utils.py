from contextlib import nullcontext
from pathlib import Path
from typing import Any, BinaryIO, ContextManager, IO, Optional, Union

import dill


def file_context(
        file: Union[IO, str, Path],
        mode: Optional[str] = None
) -> ContextManager[IO]:
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
        return open(file, mode)
    return nullcontext(file)


def safe_dump(
        obj: Any,
        file: Union[BinaryIO, str, Path]
) -> None:
    """Dumps object to file using dill.

    Args:
        obj (Any): Any object that can be dumped with dill.
        file (Union[BinaryIO, str, Path]): File to dump to.
            Either a path to the file or an open file handle.
    """
    with file_context(file, 'wb') as f:
        dill.dump(
            obj,
            file=f,
            protocol=dill.HIGHEST_PROTOCOL,
            recurse=True
        )


def safe_load(file: Union[BinaryIO, str, Path]) -> Any:
    """Loads object from file using dill.

    Args:
        file (Union[BinaryIO, str, Path]): File to load from.
            Either a path to the file or an open file handle.

    Returns:
        Any: Object loaded from the file.
    """
    with file_context(file, 'rb') as f:
        return dill.load(f)
