from __future__ import annotations

import math
from numbers import Real
from typing import ClassVar

from attr import define, field
from graphviz import Digraph


def _value_convert(value: Value | Real):
    if isinstance(value, Value):
        return value.val
    return value


def _operator_convert(op: Operator):
    if op is None:
        return None
    return op()


@define(slots=False)
class Value:
    val = field(converter=_value_convert)
    _children: set[Value] = field(default=set(), converter=set)
    _op = field(default=None, converter=_operator_convert)
    grad_ = field(default=0)

    def __add__(self, other: Value | int):
        if isinstance(other, Value):
            return Value(self.val + other.val, (self, other), AddOperator)
        return self + Value(other)

    def __radd__(self, other: int):
        return self + other

    def __neg__(self):
        return self * -1

    def __sub__(self, other: Value | Real):
        if isinstance(other, Value):
            return Value(self.val - other.val, (self, other), SubOperator)
        return self - Value(other)

    def __rsub__(self, other: Real):
        return -self + other

    def __mul__(self, other: Value | Real):
        if isinstance(other, Value):
            return Value(self.val * other.val, (self, other), MulOperator)
        return self * Value(other)

    def __rmul__(self, other: Real):
        return self * other

    def __truediv__(self, other: Value | Real):
        if isinstance(other, Value):
            return Value(self.val / other.val, (self, other), DivOperator)
        return self / Value(other)

    def __rtruediv__(self, other: Real):
        return Value(other) / self

    def __floordiv__(self, other: Value | Real):
        if isinstance(other, Value):
            return Value(self.val // other.val, (self, other))
        return self // Value(other)

    def __rfloordiv__(self, other: Real):
        return Value(other) // self

    def __pow__(self, other: Value | Real):
        if isinstance(other, Value):
            return Value(self.val**other.val, (self, other), PowOperator)
        return self ** Value(other)

    def __rpow__(self, other: Real):
        return Value(other) ** self

    def __repr__(self) -> str:
        return f"Value({self.val}{f', {self._op}' if self._op is not None else ''})"

    def __eq__(self, other: Value | Real):
        return self.val == _value_convert(other)

    def __hash__(self) -> int:
        return hash(self.val)

    def _backward(self):
        if not len(self._children):
            return

        if len(self._children) == 1:
            c1 = c2 = next(iter(self._children))
        else:
            c1, c2 = self._children

        self._op(c1, c2, self.grad_)

        c1._backward()
        c2._backward()

    def zero_(self):
        self.grad_ = 0
        for child in self._children:
            child.zero_()

    def backward(self):
        self.grad_ = 1
        self._backward()

    def _build_graph(self):
        nodes: dict[str, Value] = {}
        edges: list[tuple[Value, Value]] = list()

        def build(v: Value):
            if str(id(v)) not in nodes:
                nodes[f"{id(v)}"] = v
                for child in v._children:
                    edges.append((child, v))
                    build(child)

        build(self)
        return list(nodes.values()), edges

    def plot(self):
        digraph = Digraph(format="svg", graph_attr={"rankdir": "LR"})

        nodes, edges = self._build_graph()

        for node in nodes:
            uid = str(id(node))
            digraph.node(
                uid, f"{node.val:<.4f} | grad={node.grad_:<.4f}", shape="record"
            )

            if node._op:
                digraph.node(uid + node._op.operator, str(node._op.operator))

                digraph.edge(uid + node._op.operator, uid)

        for e1, e2 in edges:
            digraph.edge(str(id(e1)), str(id(e2)) + e2._op.operator)

        return digraph


@define(slots=False)
class Operator:
    name: ClassVar[str]
    operator: ClassVar[str]

    def __add__(self, other):
        return other + type(self).__name__

    def __radd__(self, other):
        return type(self).__name__ + other

    def __str__(self):
        return self.name


class AddOperator(Operator):
    name = "add"
    operator = "+"

    def __call__(self, a: Value, b: Value, chain_grad: Real):
        a.grad_ += chain_grad
        b.grad_ += chain_grad


class SubOperator(AddOperator):
    name = "sub"
    operator = "-"


class MulOperator(Operator):
    name = "mul"
    operator = "*"

    def __call__(self, a: Value, b: Value, chain_grad: Real):
        u, v = a.val, b.val
        a.grad_ += v * chain_grad
        b.grad_ += u * chain_grad


class DivOperator(Operator):
    name = "div"
    operator = "/"

    def __call__(self, a: Value, b: Value, chain_grad: Real):
        u, v = a.val, b.val
        a.grad_ += (1 / v) * chain_grad
        b.grad_ += (-u / v**2) * chain_grad


class PowOperator(Operator):
    name = "pow"
    operator = "^"

    def __call__(self, a: Value, b: Value, chain_grad: Real):
        u, v = a.val, b.val
        a.grad_ += (v * u ** (v - 1)) * chain_grad
        b.grad_ += (u**v * math.log(u)) * chain_grad
