from .operator_detect import detect_operator
from .split_and import split_AND
from .split_or import split_OR
from .split_implies import split_IMPLIES
from .split_xor import split_XOR
from .split_iff import split_IFF
from .split_not import split_NOT
from .split_nor import split_NOR

__all__ = [
	"detect_operator",
	"split_AND",
	"split_OR",
	"split_IMPLIES",
	"split_XOR",
	"split_IFF",
	"split_NOT",
	"split_NOR",
]