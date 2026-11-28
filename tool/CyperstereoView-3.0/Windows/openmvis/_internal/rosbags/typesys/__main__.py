# Copyright 2020-2026 Ternaris
# SPDX-License-Identifier: Apache-2.0
"""Tool to update builtin types shipped with rosbags."""

from __future__ import annotations

from itertools import groupby
from os import walk
from pathlib import Path
from typing import TYPE_CHECKING

from rosbags.typesys.store import Typestore
from rosbags.typesys.stores import Stores, get_typestore

from .codegen import generate_python_code
from .idl import get_types_from_idl
from .msg import get_types_from_msg

if TYPE_CHECKING:
    from rosbags.interfaces.typing import Typesdict


def generate_api(name: str) -> str:
    """Generate api docs."""
    return '\n'.join(
        [
            f'rosbags.typesys.stores.{name}',
            '=' * (23 + len(name)),
            '',
            f'.. automodule:: rosbags.typesys.stores.{name}',
            '   :members:',
            '   :show-inheritance:',
        ]
    )


def generate_docs(name: str, typs: Typesdict) -> str:
    """Generate types documentation."""
    res = [name, '=' * len(name)]
    for namespace, msgs in groupby([x.split('/msg/') for x in typs], key=lambda x: x[0]):
        res.append(namespace)
        res.append('*' * len(namespace))

        for _, msg in msgs:
            res.append(
                f'- :py:class:`{msg} <rosbags.typesys.stores.{name}.{namespace}__msg__{msg}>`',
            )
        res.append('')
    return '\n'.join(res)


def get_types_from_path(path: Path) -> Typesdict:
    """Walk path and parse all found messages into a typesdict."""
    typs: Typesdict = {}
    for root, dirnames, files in walk(path):
        if '.rosbags_ignore' in files:
            dirnames.clear()
            continue
        for fname in files:
            path = Path(root, fname)
            if path.suffix == '.idl':
                typs.update(get_types_from_idl(path.read_text(encoding='utf-8')))
            elif path.suffix == '.msg':
                name = path.relative_to(path.parents[2]).with_suffix('')
                if '/msg/' not in str(name):
                    name = name.parent / 'msg' / name.name
                typs.update(get_types_from_msg(path.read_text(encoding='utf-8'), str(name)))

    return dict(sorted(typs.items()))


def main(base: str, target: str, pathstr: str) -> None:  # pragma: no cover
    """Update builtin types.

    Discover message definitions in filesystem and generate types.py module.

    """
    selfdir = Path(__file__).parent

    typs = get_types_from_path(Path(pathstr))
    if not base:
        typs = {k: v for k, v in typs.items() if k.startswith('builtin')}

    typestore = get_typestore(Stores(base)) if base else Typestore()
    newstore = Typestore()
    newstore.register(typs)

    remove = sorted(typestore.fielddefs.keys() - newstore.fielddefs.keys())
    add = sorted(newstore.fielddefs.keys() - typestore.fielddefs.keys())
    change = sorted(newstore.fielddefs.keys() & typestore.fielddefs.keys())
    keep: list[str] = []

    for typ in change.copy():
        if typestore.fielddefs[typ] == newstore.fielddefs[typ]:
            keep.append(typ)
            change.remove(typ)

    Typestore().register(typs)
    _ = (
        (selfdir / 'stores' / target)
        .with_suffix('.py')
        .write_text(generate_python_code(typs, base, remove, add, change, keep))
    )
    docsdir = selfdir.parent.parent.parent / 'docs'
    _ = (docsdir / 'api' / 'stores' / f'{target}.rst').write_text(generate_api(target))
    _ = (docsdir / 'topics' / f'typesys-types-{target}.rst').write_text(generate_docs(target, typs))


if __name__ == '__main__':
    import sys

    main(*sys.argv[1:])
