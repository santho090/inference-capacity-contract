# Publication policy

The public repository is a clean core. It must not contain:

- internal hostnames, private repository URLs, credentials, customer names, or fleet identifiers;
- proprietary GPU prices, internal benchmark results, or private model recipes;
- deployment manifests tied to one organization; or
- claims that cannot be reproduced from public inputs and the pinned implementation.

Private deployments can add model catalogs, cloud prices, GPU discovery,
measured profiles, and policy adapters in separate packages. Those adapters
should attach `EvidenceRecord` sources and preserve the public contract schema.

Before a release, verify:

1. after installing the declared build backend, the package builds from a clean
   checkout with `PIP_NO_INDEX=1` and no build isolation;
2. tests and the CLI run from the source tree and installed wheel;
3. the root license and security policy are present;
4. fixtures contain no private identifiers; and
5. analytical, initialization, and SLO claims are not conflated.
