import hashlib


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_fileobj(file_obj) -> str:
    hasher = hashlib.sha256()
    current_position = file_obj.tell() if hasattr(file_obj, "tell") else None
    if hasattr(file_obj, "chunks"):
        for chunk in file_obj.chunks():
            hasher.update(chunk)
    else:
        while True:
            chunk = file_obj.read(8192)
            if not chunk:
                break
            hasher.update(chunk)
    if current_position is not None and hasattr(file_obj, "seek"):
        file_obj.seek(current_position)
    return hasher.hexdigest()
