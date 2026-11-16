"""The fast sampler has one RNG stream for the complete candidate batch."""
import secrets


SCHEME = "torch-gumbel-batch-v1"
MAX_SEED = 2**31 - 1


def job_seed(value=None):
    if value is None:
        return secrets.randbelow(MAX_SEED + 1)
    if type(value) is not int or not 0 <= value <= MAX_SEED:
        raise ValueError("job seed must be an integer in 0..2147483647")
    return value


def candidate_budget(job, reward, configured):
    # Internal comparisons request an explicit budget; chat retains best-of-N.
    requested = job.get("n", 20)
    if type(requested) is not int or not 1 <= requested <= 64:
        raise ValueError("candidate budget must be an integer in 1..64")
    return requested if job.get("source") == "admin_grid" or not reward else configured
