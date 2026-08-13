# Security policy

This library has no runtime credentials, model payload downloader, or
Kubernetes actuator. Its optional Hugging Face resolver makes unauthenticated
HTTPS requests only to `huggingface.co` for a pinned revision's JSON metadata
and bounded SafeTensors header ranges. It does not accept or store access
tokens. Callers that need private repositories should fetch metadata through
their own authenticated client and pass the resulting objects to the pure
import function.

Do not add private endpoints, tokens, fleet inventory, customer data, or
internal deployment recipes to issues, fixtures, tests, caches, or pull
requests. Treat imported values, manifests, initialization profiles, and
benchmark outputs as potentially sensitive before publishing them.

Private vulnerability reporting is not currently enabled for this repository.
Do not publish credentials or sensitive infrastructure details in a public
issue. Contact the maintainer through a private channel listed on their GitHub
profile; if none is available, withhold sensitive details until private
reporting is enabled.
