"""Data models for branch detection."""

from dataclasses import dataclass, field


@dataclass
class BranchOption:
    """
    A single option at a branch point.

    Attributes:
        id: Unique identifier for this option.
        label: The text label/description of this option.
        span: The original text span where this option was found.
        meta: Optional metadata dictionary.

    Examples:
        >>> opt = BranchOption(
        ...     id="opt1",
        ...     label="Flask",
        ...     span="1. Flask - lightweight web framework"
        ... )
    """

    id: str
    label: str
    span: str
    meta: dict[str, any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate option after initialization."""
        if not self.id:
            raise ValueError("BranchOption id cannot be empty")
        if not self.label:
            raise ValueError("BranchOption label cannot be empty")


@dataclass
class BranchPoint:
    """
    A decision point in text with multiple mutually-exclusive options.

    Attributes:
        id: Unique identifier for this branch point.
        type: Type of branch point ('enumeration', 'disjunction', 'conditional').
        options: List of available options at this branch point.
        depends_on: Optional list of branch point IDs this depends on (for nested branches).
        context: Optional context text surrounding the branch point.
        meta: Optional metadata dictionary.

    Examples:
        >>> bp = BranchPoint(
        ...     id="bp1",
        ...     type="enumeration",
        ...     options=[
        ...         BranchOption(id="opt1", label="Flask", span="1. Flask"),
        ...         BranchOption(id="opt2", label="FastAPI", span="2. FastAPI"),
        ...     ]
        ... )
        >>> len(bp.options)
        2
    """

    id: str
    type: str
    options: list[BranchOption]
    depends_on: list[str] = field(default_factory=list)
    context: str = ""
    meta: dict[str, any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate branch point after initialization."""
        if not self.id:
            raise ValueError("BranchPoint id cannot be empty")

        valid_types = ["enumeration", "disjunction", "conditional", "open_directive"]
        if self.type not in valid_types:
            raise ValueError(
                f"BranchPoint type must be one of {valid_types}, got '{self.type}'"
            )

        if len(self.options) < 2:
            raise ValueError(
                f"BranchPoint must have at least 2 options, got {len(self.options)}"
            )

        # Validate all options are BranchOption instances
        for opt in self.options:
            if not isinstance(opt, BranchOption):
                raise TypeError(
                    f"All options must be BranchOption instances, got {type(opt)}"
                )

    @property
    def option_count(self) -> int:
        """Return the number of options at this branch point."""
        return len(self.options)

    def get_option_labels(self) -> list[str]:
        """Return list of option labels."""
        return [opt.label for opt in self.options]
