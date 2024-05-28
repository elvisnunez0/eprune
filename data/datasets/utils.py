import pickle


def unpickle(file: str) -> dict:
    """
    Unpickles the file at the provided path.

    Args:
        file: The path to the file to unpickle.

    Returns:
        dict: The unpickled file.
    """
    with open(file, "rb") as fo:
        unpickleed_file = pickle.load(fo, encoding="bytes")
    return unpickleed_file
