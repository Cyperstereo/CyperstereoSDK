# Copyright 2020-2026 Ternaris
# SPDX-License-Identifier: Apache-2.0
"""Type store."""

from __future__ import annotations

import json
import struct
import sys
from hashlib import md5, sha256
from importlib.util import module_from_spec, spec_from_loader
from itertools import starmap
from struct import pack_into
from typing import TYPE_CHECKING, Protocol, TypeVar, cast

from rosbags.interfaces import Msgdef, Nodetype
from rosbags.serde import SerdeError
from rosbags.serde.cdr import generate_deserialize_cdr, generate_getsize_cdr, generate_serialize_cdr
from rosbags.serde.ros1 import (
    generate_cdr_to_ros1,
    generate_deserialize_ros1,
    generate_getsize_ros1,
    generate_ros1_to_cdr,
    generate_serialize_ros1,
)
from rosbags.typesys.base import TypesysError, normalize_fieldname

from .codegen import generate_python_code
from .msg import denormalize_msgtype

if TYPE_CHECKING:
    from types import ModuleType
    from typing import ClassVar, TypeAlias, TypedDict

    from numpy.typing import ArrayLike

    from rosbags.interfaces import Bitcvt, BitcvtSize
    from rosbags.interfaces.typing import (
        FieldDesc,
        Typesdict,
    )

    class FieldType(TypedDict):
        """Field type."""

        type_id: int
        capacity: int
        string_capacity: int
        nested_type_name: str

    class Field(TypedDict):
        """Field."""

        name: str
        type: FieldType

    class Struct(TypedDict):
        """Struct."""

        type_name: str
        fields: list[Field]

    class MsgType(Protocol):
        """AnyInit."""

        __msgtype__: ClassVar[str]

        def __init__(self, *args: Msgarg, **kwargs: Msgarg) -> None:
            """Accept any args."""
            raise NotImplementedError

    class Msg(Protocol):
        """AnyInit."""

        __msgtype__: ClassVar[str]

    Msgarg: TypeAlias = Msg | list[Msg] | bool | int | float | str | list[str] | ArrayLike


T = TypeVar('T')

TIDMAP = {
    'int8': 2,
    'uint8': 3,
    'int16': 4,
    'uint16': 5,
    'int32': 6,
    'uint32': 7,
    'int64': 8,
    'uint64': 9,
    'float32': 10,
    'float64': 11,
    'float128': 12,
    'char': 13,
    # Unsupported 'wchar': 14,
    'bool': 15,
    'byte': 16,
    'string': 17,
    # Unsupported 'wstring': 18,
    # Unsupported 'fixed_string': 19,
    # Unsupported 'fixed_wstring': 20,
    'bounded_string': 21,
    # Unsupported 'bounded_wstring': 22,
}


class Typestore:
    """Type store."""

    types: dict[str, type[MsgType]]
    fielddefs: Typesdict

    def __init__(
        self,
        base: ModuleType | None = None,
        *,
        max_sequence_length: int = 2**32 - 1,
    ) -> None:
        """Initialize."""
        if max_sequence_length < 1:
            msg = 'max_sequence_length must be positive.'
            raise ValueError(msg)
        self.cache: dict[str, Msgdef[object]] = {}
        self.types = {}
        self.fielddefs = {}
        if base:
            fielddefs = cast('Typesdict', base.FIELDDEFS)
            self.fielddefs.update(fielddefs)
            self.types.update({k: getattr(base, k.replace('/', '__')) for k in fielddefs})
        self.max_sequence_length = max_sequence_length

    def deserialize_cdr(self, rawdata: bytes | memoryview, typename: str) -> object:
        """Deserialize raw data into a message object.

        Args:
            rawdata: Serialized data.
            typename: Message type name.

        Returns:
            Deserialized message object.

        """
        if len(rawdata) < 4 or bytes(rawdata[:2]) not in {b'\x00\x00', b'\x00\x01'}:
            msg = 'Invalid or unsupported CDR encapsulation.'
            raise SerdeError(msg)
        little_endian = bool(rawdata[1])

        msgdef = self.get_msgdef(typename)
        func = msgdef.deserialize_cdr_le if little_endian else msgdef.deserialize_cdr_be
        try:
            message, pos = func(rawdata[4:], 0, msgdef.cls, self)
        except (struct.error, UnicodeDecodeError, ValueError) as exc:
            msg = f'Could not deserialize {typename!r}: {exc}'
            raise SerdeError(msg) from exc
        if pos + 4 + 3 < len(rawdata):
            msg = f'CDR message size mismatch for {typename!r}.'
            raise SerdeError(msg)
        return message

    def serialize_cdr(
        self,
        message: object,
        typename: str,
        *,
        little_endian: bool = sys.byteorder == 'little',
    ) -> memoryview:
        """Serialize message object to bytes.

        Args:
            message: Message object.
            typename: Message type name.
            little_endian: Should use little endianness.
            typestore: Type store.

        Returns:
            Serialized bytes.

        """
        msgdef = self.get_msgdef(typename)
        size = 4 + msgdef.getsize_cdr(0, message, self)
        rawdata = memoryview(bytearray(size))
        pack_into('BB', rawdata, 0, 0, little_endian)

        func = msgdef.serialize_cdr_le if little_endian else msgdef.serialize_cdr_be

        pos = func(rawdata[4:], 0, message, self)
        assert pos + 4 == size
        return rawdata.toreadonly()

    def deserialize_ros1(self, rawdata: bytes | memoryview, typename: str) -> object:
        """Deserialize raw data into a message object.

        Args:
            rawdata: Serialized data.
            typename: Message type name.
            typestore: Type store.

        Returns:
            Deserialized message object.

        """
        msgdef = self.get_msgdef(typename)
        func = msgdef.deserialize_ros1
        try:
            message, pos = func(rawdata, 0, msgdef.cls, self)
        except (struct.error, UnicodeDecodeError, ValueError) as exc:
            msg = f'Could not deserialize {typename!r}: {exc}'
            raise SerdeError(msg) from exc
        if pos != len(rawdata):
            msg = f'ROS1 message size mismatch for {typename!r}.'
            raise SerdeError(msg)
        return message

    def serialize_ros1(self, message: object, typename: str) -> memoryview:
        """Serialize message object to bytes.

        Args:
            message: Message object.
            typename: Message type name.
            typestore: Type store.

        Returns:
            Serialized bytes.

        """
        msgdef = self.get_msgdef(typename)
        size = msgdef.getsize_ros1(0, message, self)
        rawdata = memoryview(bytearray(size))
        func = msgdef.serialize_ros1
        pos = func(rawdata, 0, message, self)
        assert pos == size
        return rawdata.toreadonly()

    def ros1_to_cdr(self, raw: bytes | memoryview, typename: str) -> memoryview:
        """Convert serialized ROS1 message directly to CDR.

        This should be reasonably fast as conversions happen on a byte-level
        without going through deserialization and serialization.

        Args:
            raw: ROS1 serialized message.
            typename: Message type name.
            typestore: Type store.

        Returns:
            CDR serialized message.

        """
        msgdef = self.get_msgdef(typename)

        try:
            ipos, opos = msgdef.getsize_ros1_to_cdr(raw, 0, None, 0, self)
        except (struct.error, ValueError) as exc:
            msg = f'Could not convert {typename!r}: {exc}'
            raise SerdeError(msg) from exc
        if ipos != len(raw):
            msg = f'ROS1 message size mismatch for {typename!r}.'
            raise SerdeError(msg)

        size = 4 + opos
        rawdata = memoryview(bytearray(size))
        pack_into('BB', rawdata, 0, 0, 1)

        ipos, opos = msgdef.ros1_to_cdr(raw, 0, rawdata[4:], 0, self)
        assert ipos == len(raw)
        assert opos + 4 == size
        return rawdata.toreadonly()

    def cdr_to_ros1(self, raw: bytes | memoryview, typename: str) -> memoryview:
        """Convert serialized CDR message directly to ROS1.

        This should be reasonably fast as conversions happen on a byte-level
        without going through deserialization and serialization.

        Args:
            raw: CDR serialized message.
            typename: Message type name.
            typestore: Type store.

        Returns:
            ROS1 serialized message.

        """
        if len(raw) < 4 or bytes(raw[:2]) != b'\x00\x01':
            msg = 'Direct CDR conversion requires little-endian CDR encapsulation.'
            raise SerdeError(msg)

        msgdef = self.get_msgdef(typename)

        try:
            ipos, opos = msgdef.getsize_cdr_to_ros1(raw[4:], 0, None, 0, self)
        except (struct.error, ValueError) as exc:
            msg = f'Could not convert {typename!r}: {exc}'
            raise SerdeError(msg) from exc
        if ipos + 4 + 3 < len(raw):
            msg = f'CDR message size mismatch for {typename!r}.'
            raise SerdeError(msg)

        size = opos
        rawdata = memoryview(bytearray(size))

        ipos, opos = msgdef.cdr_to_ros1(raw[4:], 0, rawdata, 0, self)
        assert ipos + 4 + 3 >= len(raw)
        assert opos == size
        return rawdata.toreadonly()

    def register(self, typs: Typesdict) -> None:
        """Register types in the store.

        Args:
            typs: Dictionary mapping message typenames to parsetrees.

        Raises:
            TypesysError: Type already present with different definition.

        """
        code = generate_python_code(typs)
        name = 'rosbags.usertypes'
        spec = spec_from_loader(name, loader=None)
        assert spec
        module = module_from_spec(spec)
        sys.modules[name] = module
        exec(code, module.__dict__)  # noqa: S102
        fielddefs = cast('Typesdict', module.FIELDDEFS)

        for name, (_, fields) in fielddefs.items():
            if have := self.fielddefs.get(name):
                _, have_fields = have
                have_fields = [(x[0].lower(), x[1]) for x in have_fields]
                new_fields = [(x[0].lower(), x[1]) for x in fields]
                if have_fields != new_fields:
                    msg = f'Type {name!r} is already present with different definition.'
                    raise TypesysError(msg)

        for name in fielddefs.keys() - self.fielddefs.keys():
            self.fielddefs[name] = fielddefs[name][:2]
            self.types[name] = getattr(module, name.replace('/', '__'))

    def get_msgdef(self, typename: str) -> Msgdef[object]:
        """Retrieve message definition for typename.

        Message definitions are cached globally and generated as needed.

        Args:
            typename: Msgdef type name to load.
            typestore: Type store.

        Returns:
            Message definition.

        """
        if typename not in self.cache:
            entries = [
                (normalize_fieldname(name), desc) for name, desc in self.fielddefs[typename][1]
            ]

            getsize_cdr, size_cdr = generate_getsize_cdr(entries, self)
            getsize_ros1, size_ros1 = generate_getsize_ros1(entries, self)

            self.cache[typename] = Msgdef(
                typename,
                entries,
                self.types[typename],
                size_cdr,
                getsize_cdr,
                generate_serialize_cdr(entries, self, 'le'),
                generate_serialize_cdr(entries, self, 'be'),
                generate_deserialize_cdr(entries, self, 'le'),
                generate_deserialize_cdr(entries, self, 'be'),
                size_ros1,
                getsize_ros1,
                generate_serialize_ros1(entries, self),
                generate_deserialize_ros1(entries, self),
                cast('BitcvtSize', generate_ros1_to_cdr(entries, typename, self, copy=False)),
                cast('Bitcvt', generate_ros1_to_cdr(entries, typename, self, copy=True)),
                cast('BitcvtSize', generate_cdr_to_ros1(entries, typename, self, copy=False)),
                cast('Bitcvt', generate_cdr_to_ros1(entries, typename, self, copy=True)),
            )
        return self.cache[typename]

    def gendefhash(
        self,
        typename: str,
        subdefs: dict[str, tuple[str, str]],
        ros_version: int = 1,
    ) -> tuple[str, str]:
        """Generate message definition and hash for type.

        The subdefs argument will be filled with child definitions.

        Args:
            typename: Name of type to generate definition for.
            subdefs: Child definitions.
            ros_version: ROS version number.

        Returns:
            Message definition and hash.

        Raises:
            TypesysError: Type does not exist.

        """
        typemap = (
            {'builtin_interfaces/msg/Time': 'time', 'builtin_interfaces/msg/Duration': 'duration'}
            if ros_version == 1
            else {}
        )

        deftext: list[str] = []
        hashtext: list[str] = []
        if typename not in self.fielddefs:
            msg = f'Type {typename!r} is unknown.'
            raise TypesysError(msg)

        for name, typ, value in self.fielddefs[typename][0]:
            deftext.append(f'{typ} {name}={value}')
            hashtext.append(f'{typ} {name}={value}')

        for name, desc in self.fielddefs[typename][1]:
            if name == 'structure_needs_at_least_one_member':
                continue
            if desc[0] == Nodetype.BASE:
                argname: str
                argname, arglimit = desc[1]
                if argname == 'string':
                    argname = f'string<={arglimit}' if arglimit else 'string'
                deftext.append(f'{argname} {name}')
                hashtext.append(f'{argname} {name}')
            elif desc[0] == int(Nodetype.NAME):
                args = desc[1]
                assert isinstance(args, str)
                subname = args
                if subname in typemap:
                    deftext.append(f'{typemap[subname]} {name}')
                    hashtext.append(f'{typemap[subname]} {name}')
                else:
                    if subname not in subdefs:
                        subdefs[subname] = ('', '')
                        subdefs[subname] = self.gendefhash(subname, subdefs, ros_version)
                    deftext.append(f'{denormalize_msgtype(subname)} {name}')
                    hashtext.append(f'{subdefs[subname][1]} {name}')
            else:
                assert desc[0] in {Nodetype.ARRAY, Nodetype.SEQUENCE}
                assert isinstance(desc[1], tuple)
                subdesc, num = desc[1]
                isubname: tuple[str, int] | str
                isubtype, isubname = subdesc
                count = (
                    '' if num == 0 else str(num) if desc[0] == int(Nodetype.ARRAY) else f'<={num}'
                )
                if isubtype == int(Nodetype.BASE):
                    if isubname[0] == 'string':
                        isubname = (f'string<={isubname[1]}' if isubname[1] else 'string', 0)
                    deftext.append(f'{isubname[0]}[{count}] {name}')
                    hashtext.append(f'{isubname[0]}[{count}] {name}')
                elif isubname in typemap:
                    assert isinstance(isubname, str)
                    deftext.append(f'{typemap[isubname]}[{count}] {name}')
                    hashtext.append(f'{typemap[isubname]}[{count}] {name}')
                else:
                    assert isinstance(isubname, str)
                    if isubname not in subdefs:
                        subdefs[isubname] = ('', '')
                        subdefs[isubname] = self.gendefhash(isubname, subdefs, ros_version)
                    deftext.append(f'{denormalize_msgtype(isubname)}[{count}] {name}')
                    hashtext.append(f'{subdefs[isubname][1]} {name}')

        deftext.append('')
        return '\n'.join(deftext), md5('\n'.join(hashtext).encode()).hexdigest()  # noqa: S324

    def generate_msgdef(
        self,
        typename: str,
        ros_version: int = 1,
    ) -> tuple[str, str]:
        """Generate message definition for type.

        Args:
            typename: Name of type to generate definition for.
            typestore: Custom type store.
            ros_version: ROS version number.

        Returns:
            Message definition.

        """
        subdefs: dict[str, tuple[str, str]] = {}
        msgdef, md5sum = self.gendefhash(typename, subdefs, ros_version)

        msgdef = ''.join(
            [
                msgdef,
                *[f'{"=" * 80}\nMSG: {denormalize_msgtype(k)}\n{v[0]}' for k, v in subdefs.items()],
            ],
        )

        return msgdef, md5sum

    def hash_rihs01(self, typ: str) -> str:
        """Hash message definition.

        Args:
            typ: Message type name.

        Returns:
            Hash value.

        """

        def get_field(name: str, desc: FieldDesc) -> Field:
            increment = 0
            capacity = 0
            string_capacity = 0
            subtype = ''
            if desc[0] == Nodetype.ARRAY:
                increment = 48
                capacity = desc[1][1]
                typ, rest = desc[1][0]
            elif desc[0] == Nodetype.SEQUENCE:
                count = desc[1][1]
                if count:
                    increment = 96
                    capacity = count
                else:
                    increment = 144
                typ, rest = desc[1][0]
            else:
                typ, rest = desc

            if typ == Nodetype.NAME:
                tid = increment + 1
                assert isinstance(rest, str)
                subtype = rest
                _ = get_struct(subtype)
            elif rest[0] == 'string' and rest[1]:
                assert isinstance(rest[1], int)
                string_capacity = rest[1]
                tid = increment + TIDMAP['bounded_string']
            else:
                assert isinstance(rest[0], str)
                tid = increment + TIDMAP[rest[0]]

            return {
                'name': name,
                'type': {
                    'type_id': tid,
                    'capacity': capacity,
                    'string_capacity': string_capacity,
                    'nested_type_name': subtype,
                },
            }

        struct_cache: dict[str, Struct] = {}

        def get_struct(typ: str) -> Struct:
            if typ not in struct_cache:
                struct_cache[typ] = {
                    'type_name': typ,
                    'fields': list(
                        starmap(
                            get_field,
                            self.fielddefs[typ][1]
                            or [
                                (
                                    'structure_needs_at_least_one_member',
                                    (Nodetype.BASE, ('uint8', 0)),
                                )
                            ],
                        )
                    ),
                }
            return struct_cache[typ]

        dct = {
            'type_description': get_struct(typ),
            'referenced_type_descriptions': [
                y for x, y in sorted(struct_cache.items()) if x != typ
            ],
        }

        digest = sha256(json.dumps(dct).encode()).hexdigest()
        return f'RIHS01_{digest}'
