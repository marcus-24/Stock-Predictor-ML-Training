class InvalidBranchNameError(Exception):

    def __init__(self, value: str, message: str) -> None:
        self._value = value
        self._message = message
        super().__init__(message)


def set_env_name(branch_name: str) -> str:
    match branch_name:
        case "local_branch":
            return "personal computer"
        case "develop":
            return "dev"
        case "main":
            return "production"
        case _:
            raise InvalidBranchNameError(
                value=branch_name,
                message="An invalid branch name was used to trigger the machine learning training pipeline",
            )
