def separate(audio_path: str, job_id: str, params: dict, out_dir: str) -> str:
    # Thin wrapper that would call your Modal function (GPU)
    # For now, just import the local function to keep same signature
    from .nemo_separator import separate as local_sep
    return local_sep(audio_path, job_id, params, out_dir)
