from .operator_detect import detect_operator
from .split_and import split_AND
from .split_or import split_OR
from .split_implies import split_IMPLIES
from .split_xor import split_XOR
from .split_iff import split_IFF

__all__ = [
	"detect_operator",
	"split_AND",
	"split_OR",
	"split_IMPLIES",
	"split_XOR",
	"split_IFF",
]
